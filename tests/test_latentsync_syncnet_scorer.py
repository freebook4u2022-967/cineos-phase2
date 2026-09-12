from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from cineos.atlas.latentsync_syncnet_scorer import (
    LATENTSYNC_PINNED_REVISION,
    LatentSyncSyncNetError,
    LatentSyncSyncNetScorer,
    latentsync_syncnet_component,
)


class _Completed:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


def _scorer(tmp_path: Path) -> tuple[LatentSyncSyncNetScorer, Path, Path]:
    repo = tmp_path / "LatentSync"
    repo.mkdir()
    checkpoint = tmp_path / "syncnet.model"
    checkpoint.write_bytes(b"real-checkpoint-fixture")
    artifact = tmp_path / "shot.mp4"
    artifact.write_bytes(b"av-artifact")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    scorer = LatentSyncSyncNetScorer(
        repository_root=repo,
        checkpoint_path=checkpoint,
        checkpoint_sha256=digest,
        minimum_confidence=3.0,
        maximum_abs_offset_frames=2,
    )
    return scorer, repo, artifact


def test_measured_av_sync_passes_only_when_confidence_and_offset_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scorer, repo, artifact = _scorer(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if command[:3] == ["git", "-C", str(repo)]:
            return _Completed(stdout=LATENTSYNC_PINNED_REVISION + "\n")
        assert kwargs["cwd"] != repo
        assert str(repo) in kwargs["env"]["PYTHONPATH"]
        return _Completed(stdout="SyncNet confidence: 4.25\nAV offset: -1\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = scorer(None, artifact=artifact, shot=object(), attempt_index=0)

    assert result == {"dialogue_lip_sync": 1.0}
    assert scorer.last_measurement == {
        "syncnet_confidence": 4.25,
        "av_offset_frames": -1,
    }
    assert len(calls) == 2


def test_measured_av_sync_rejects_large_offset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scorer, repo, artifact = _scorer(tmp_path)

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "-C", str(repo)]:
            return _Completed(stdout=LATENTSYNC_PINNED_REVISION + "\n")
        return _Completed(stdout="SyncNet confidence: 5.10\nAV offset: 4\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert scorer(None, artifact=artifact, shot=object(), attempt_index=0) == {
        "dialogue_lip_sync": 0.0
    }


def test_runtime_verification_fails_closed_on_wrong_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scorer, repo, artifact = _scorer(tmp_path)

    def fake_run(command, **kwargs):
        return _Completed(stdout="0" * 40 + "\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LatentSyncSyncNetError, match="revision"):
        scorer(None, artifact=artifact, shot=object(), attempt_index=0)


def test_runtime_verification_fails_closed_on_checkpoint_substitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scorer, repo, artifact = _scorer(tmp_path)
    scorer.checkpoint_path.write_bytes(b"substituted")

    def fake_run(command, **kwargs):
        return _Completed(stdout=LATENTSYNC_PINNED_REVISION + "\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LatentSyncSyncNetError, match="SHA-256"):
        scorer(None, artifact=artifact, shot=object(), attempt_index=0)


def test_malformed_upstream_output_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scorer, repo, artifact = _scorer(tmp_path)

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "-C", str(repo)]:
            return _Completed(stdout=LATENTSYNC_PINNED_REVISION + "\n")
        return _Completed(stdout="evaluation finished without metrics")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LatentSyncSyncNetError, match="confidence and AV offset"):
        scorer(None, artifact=artifact, shot=object(), attempt_index=0)


def test_subprocess_failure_is_not_silently_converted_to_quality_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scorer, repo, artifact = _scorer(tmp_path)

    def fake_run(command, **kwargs):
        if command[:3] == ["git", "-C", str(repo)]:
            return _Completed(stdout=LATENTSYNC_PINNED_REVISION + "\n")
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(LatentSyncSyncNetError, match="evaluation failed"):
        scorer(None, artifact=artifact, shot=object(), attempt_index=0)


def test_component_declares_only_dialogue_lip_sync(tmp_path: Path) -> None:
    scorer, _, _ = _scorer(tmp_path)
    component = latentsync_syncnet_component(scorer)
    assert component.name == "latentsync_syncnet_av"
    assert component.measured_metrics == ("dialogue_lip_sync",)
    provenance = scorer.runtime_provenance()
    assert provenance["origin"] == "external_pretrained"
    assert provenance["code_license"] == "Apache-2.0"
    assert provenance["checkpoint_license"] == "OpenRAIL++"
    assert provenance["score_semantics"] == "binary_pass_fail_not_probability"
