from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from cineos.atlas.latentsync_syncnet_scorer import (
    LATENTSYNC_PINNED_REVISION,
    LatentSyncSyncNetError,
    LatentSyncSyncNetScorer,
)


class _Completed:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


class _Shot:
    def __init__(self, dialogue_timing: list[dict]) -> None:
        self.performance = {"dialogue_timing": dialogue_timing}


def _scorer(tmp_path: Path) -> tuple[LatentSyncSyncNetScorer, Path, Path]:
    repo = tmp_path / "LatentSync"
    repo.mkdir()
    checkpoint = tmp_path / "syncnet.model"
    checkpoint.write_bytes(b"real-checkpoint-fixture")
    artifact = tmp_path / "shot.mp4"
    artifact.write_bytes(b"av-artifact")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return (
        LatentSyncSyncNetScorer(
            repository_root=repo,
            checkpoint_path=checkpoint,
            checkpoint_sha256=digest,
            minimum_confidence=3.0,
            maximum_abs_offset_frames=2,
        ),
        repo,
        artifact,
    )


def _write_face_tracks(kwargs: dict, count: int) -> None:
    crop_dir = Path(kwargs["cwd"]) / "detect_results" / "crop"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (crop_dir / f"{index:05d}.mp4").write_bytes(b"face-track")


def _two_speaker_shot() -> _Shot:
    return _Shot(
        [
            {
                "speaker_id": "lead",
                "speaker_face_track_index": 0,
                "start_seconds": 0.20,
                "end_seconds": 1.10,
            },
            {
                "speaker_id": "support",
                "speaker_face_track_index": 1,
                "start_seconds": 1.25,
                "end_seconds": 2.20,
            },
        ]
    )


def test_multispeaker_dialogue_is_measured_per_speaker_cue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scorer, repo, artifact = _scorer(tmp_path)
    shot = _two_speaker_shot()
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        if command[:3] == ["git", "-C", str(repo)]:
            return _Completed(stdout=LATENTSYNC_PINNED_REVISION + "\n")
        if command[0] == "ffmpeg":
            filter_graph = command[command.index("-filter_complex") + 1]
            assert "trim=start=" in filter_graph
            assert "atrim=start=" in filter_graph
            Path(command[-1]).write_bytes(b"speaker-cue-av")
            return _Completed()
        cwd_name = Path(kwargs["cwd"]).name
        if cwd_name == "cineos-latentsync-qc-ignored":
            raise AssertionError("unexpected fixture path")
        if cwd_name.startswith("speaker-cue-eval-"):
            _write_face_tracks(kwargs, 1)
            if cwd_name.endswith("000"):
                return _Completed(stdout="SyncNet confidence: 4.80\nAV offset: 1\n")
            return _Completed(stdout="SyncNet confidence: 3.70\nAV offset: -2\n")
        _write_face_tracks(kwargs, 2)
        return _Completed(stdout="SyncNet confidence: 0.10\nAV offset: 9\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert scorer(None, artifact=artifact, shot=shot, attempt_index=0) == {
        "dialogue_lip_sync": 1.0
    }
    assert scorer.last_measurement == {
        "syncnet_confidence": 3.7,
        "av_offset_frames": -2,
        "detected_face_tracks": 2,
        "accepted_face_tracks": 1,
        "speaker_evaluation_mode": "speaker_bound_cue_windows",
        "evaluated_dialogue_cues": 2,
        "evaluated_speakers": "lead,support",
        "speaker_binding_source": (
            "performance.dialogue_timing[*].speaker_face_track_index+cue_window"
        ),
    }
    assert sum(command[0] == "ffmpeg" for command in commands) == 2
    assert sum("eval.eval_sync_conf" in command for command in commands) == 3


def test_one_bad_speaker_cue_rejects_the_whole_dialogue_shot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scorer, repo, artifact = _scorer(tmp_path)

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "-C", str(repo)]:
            return _Completed(stdout=LATENTSYNC_PINNED_REVISION + "\n")
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"speaker-cue-av")
            return _Completed()
        cwd_name = Path(kwargs["cwd"]).name
        if cwd_name.startswith("speaker-cue-eval-"):
            _write_face_tracks(kwargs, 1)
            if cwd_name.endswith("000"):
                return _Completed(stdout="SyncNet confidence: 4.20\nAV offset: 0\n")
            return _Completed(stdout="SyncNet confidence: 2.20\nAV offset: 0\n")
        _write_face_tracks(kwargs, 2)
        return _Completed(stdout="SyncNet confidence: 8.00\nAV offset: 0\n")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert scorer(
        None, artifact=artifact, shot=_two_speaker_shot(), attempt_index=0
    ) == {"dialogue_lip_sync": 0.0}
    assert scorer.last_measurement is not None
    assert scorer.last_measurement["syncnet_confidence"] == 2.2


def test_multispeaker_cue_windows_fail_closed_without_timestamps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scorer, repo, artifact = _scorer(tmp_path)
    shot = _Shot(
        [
            {"speaker_id": "lead", "speaker_face_track_index": 0},
            {"speaker_id": "support", "speaker_face_track_index": 1},
        ]
    )

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "-C", str(repo)]:
            return _Completed(stdout=LATENTSYNC_PINNED_REVISION + "\n")
        _write_face_tracks(kwargs, 2)
        return _Completed(stdout="SyncNet confidence: 5.0\nAV offset: 0\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LatentSyncSyncNetError, match="start_seconds/end_seconds"):
        scorer(None, artifact=artifact, shot=shot, attempt_index=0)


def test_multispeaker_cue_window_rejects_out_of_range_track(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scorer, repo, artifact = _scorer(tmp_path)
    shot = _Shot(
        [
            {
                "speaker_id": "lead",
                "speaker_face_track_index": 0,
                "start_seconds": 0.1,
                "end_seconds": 0.8,
            },
            {
                "speaker_id": "support",
                "speaker_face_track_index": 4,
                "start_seconds": 0.9,
                "end_seconds": 1.6,
            },
        ]
    )

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "-C", str(repo)]:
            return _Completed(stdout=LATENTSYNC_PINNED_REVISION + "\n")
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"speaker-cue-av")
            return _Completed()
        if Path(kwargs["cwd"]).name.startswith("speaker-cue-eval-"):
            _write_face_tracks(kwargs, 1)
            return _Completed(stdout="SyncNet confidence: 4.0\nAV offset: 0\n")
        _write_face_tracks(kwargs, 2)
        return _Completed(stdout="SyncNet confidence: 5.0\nAV offset: 0\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LatentSyncSyncNetError, match="out of range"):
        scorer(None, artifact=artifact, shot=shot, attempt_index=0)
