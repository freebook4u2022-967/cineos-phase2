from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cineos.audio.lipsync_qc import (
    ExternalLipSyncAnalyzer,
    LipSyncQCError,
    LipSyncQualityEvidence,
    validate_lipsync_quality_evidence,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _evidence(
    *, video_sha: str, audio_sha: str, **overrides: object
) -> dict[str, object]:
    item = LipSyncQualityEvidence(
        shot_id="shot-01",
        video_sha256=video_sha,
        audio_sha256=audio_sha,
        analyzer_origin="external_pretrained_foundation",
        analyzer_id="learned-av-sync-evaluator",
        analyzer_revision="immutable-revision-1",
        analyzer_license_id="declared-license",
        analyzer_source_url="https://example.invalid/model-card",
        sync_confidence=0.91,
        av_offset_ms=24.0,
        speaking_frame_coverage=0.88,
        face_track_coverage=0.97,
        measured=True,
        accepted=True,
    ).to_dict()
    if overrides:
        item.update(overrides)
        unsigned = dict(item)
        unsigned.pop("evidence_sha256", None)
        payload = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        item["evidence_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return item


def test_valid_measured_lipsync_evidence_is_artifact_bound() -> None:
    video_sha = "a" * 64
    audio_sha = "b" * 64
    result = validate_lipsync_quality_evidence(
        _evidence(video_sha=video_sha, audio_sha=audio_sha),
        expected_shot_id="shot-01",
        expected_video_sha256=video_sha,
        expected_audio_sha256=audio_sha,
    )
    assert result.accepted is True
    assert result.analyzer_origin == "external_pretrained_foundation"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("video_sha256", "g" * 64),
        ("audio_sha256", "z" * 64),
        ("measured", False),
        ("analyzer_origin", "cineos_native"),
    ],
)
def test_lipsync_evidence_fails_closed_on_invalid_production_claims(
    field: str, value: object
) -> None:
    video_sha = "a" * 64
    audio_sha = "b" * 64
    with pytest.raises(LipSyncQCError):
        validate_lipsync_quality_evidence(
            _evidence(
                video_sha=video_sha,
                audio_sha=audio_sha,
                **{field: value},
            ),
            expected_shot_id="shot-01",
            expected_video_sha256=video_sha,
            expected_audio_sha256=audio_sha,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"sync_confidence": 0.64, "accepted": False},
        {"av_offset_ms": 120.1, "accepted": False},
        {"speaking_frame_coverage": 0.59, "accepted": False},
        {"face_track_coverage": 0.89, "accepted": False},
    ],
)
def test_lipsync_evidence_rejects_failed_measured_thresholds(
    overrides: dict[str, object],
) -> None:
    video_sha = "a" * 64
    audio_sha = "b" * 64
    with pytest.raises(LipSyncQCError, match="failed measured lip-sync QC"):
        validate_lipsync_quality_evidence(
            _evidence(video_sha=video_sha, audio_sha=audio_sha, **overrides),
            expected_shot_id="shot-01",
            expected_video_sha256=video_sha,
            expected_audio_sha256=audio_sha,
        )


def test_lipsync_evidence_rejects_tampering() -> None:
    evidence = _evidence(video_sha="a" * 64, audio_sha="b" * 64)
    evidence["sync_confidence"] = 0.99
    with pytest.raises(LipSyncQCError, match="does not match its contents"):
        validate_lipsync_quality_evidence(
            evidence,
            expected_shot_id="shot-01",
            expected_video_sha256="a" * 64,
            expected_audio_sha256="b" * 64,
        )


def test_external_analyzer_binds_exact_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "shot.mp4"
    audio = tmp_path / "dialogue.wav"
    video.write_bytes(b"video-frames")
    audio.write_bytes(b"dialogue-audio")
    observed: dict[str, object] = {}

    class Completed:
        stdout = json.dumps(
            {
                "analyzer_id": "learned-av-sync-evaluator",
                "analyzer_revision": "immutable-revision-1",
                "analyzer_license_id": "declared-license",
                "analyzer_source_url": "https://example.invalid/model-card",
                "sync_confidence": 0.92,
                "av_offset_ms": -16.0,
                "speaking_frame_coverage": 0.82,
                "face_track_coverage": 0.96,
            }
        )

    def fake_run(argv: list[str], **kwargs: object) -> Completed:
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return Completed()

    monkeypatch.setattr("cineos.audio.lipsync_qc.subprocess.run", fake_run)
    analyzer = ExternalLipSyncAnalyzer(
        command=("sync-eval", "--video", "{video}", "--audio", "{audio}")
    )
    evidence = analyzer.measure(shot_id="shot-01", video_path=video, audio_path=audio)

    assert evidence.video_sha256 == _digest(b"video-frames")
    assert evidence.audio_sha256 == _digest(b"dialogue-audio")
    argv = observed["argv"]
    assert isinstance(argv, list)
    assert str(video.resolve()) in argv
    assert str(audio.resolve()) in argv
    assert evidence.accepted is True


def test_external_analyzer_requires_both_artifact_placeholders() -> None:
    analyzer = ExternalLipSyncAnalyzer(command=("sync-eval", "--video", "{video}"))
    with pytest.raises(LipSyncQCError, match=r"\{video\} and \{audio\}"):
        analyzer.measure(
            shot_id="shot-01",
            video_path=Path("missing.mp4"),
            audio_path=Path("missing.wav"),
        )
