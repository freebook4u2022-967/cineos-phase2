import hashlib

import pytest

from cineos.audio.production_lipsync import (
    ProductionLipSyncError,
    ProductionLipSyncPolicy,
    evidence_manifest,
    validate_production_lipsync,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts(tmp_path):
    video = tmp_path / "shot-01.mp4"
    audio = tmp_path / "cue-01.wav"
    video.write_bytes(b"rendered-shot")
    audio.write_bytes(b"approved-dialogue")
    return video, audio


def _report(video, audio, **overrides):
    report = {
        "shot_id": "shot-01",
        "character_id": "char-a",
        "dialogue_cue_id": "cue-01",
        "rendered_video_sha256": _sha(video),
        "dialogue_audio_sha256": _sha(audio),
        "scorer_name": "external-measured-av-sync",
        "scorer_provenance": "declared external/open scorer; not CINEOS-native",
        "sync_confidence": 0.92,
        "mean_offset_ms": 54.0,
        "p95_offset_ms": 96.0,
        "measured_frame_count": 120,
        "measured_word_count": 7,
    }
    report.update(overrides)
    return report


def _validate(report, video, audio, **kwargs):
    return validate_production_lipsync(
        report,
        rendered_video_path=video,
        dialogue_audio_path=audio,
        shot_id="shot-01",
        character_id="char-a",
        dialogue_cue_id="cue-01",
        **kwargs,
    )


def test_accepts_measured_artifact_bound_lipsync_evidence(tmp_path):
    video, audio = _artifacts(tmp_path)
    evidence = _validate(_report(video, audio), video, audio)

    assert evidence.accepted is True
    assert evidence.decision == "accept"
    assert evidence.failed_metrics == ()
    assert evidence.rendered_video_sha256 == _sha(video)
    assert evidence.dialogue_audio_sha256 == _sha(audio)
    assert len(evidence.evidence_sha256) == 64
    assert (
        evidence_manifest(evidence)["schema"]
        == "cineos-production-lipsync-evidence/0.1"
    )


def test_rejects_stale_or_swapped_artifact_binding(tmp_path):
    video, audio = _artifacts(tmp_path)
    other_video = tmp_path / "other.mp4"
    other_video.write_bytes(b"different-render")

    report = _report(video, audio, rendered_video_sha256=_sha(other_video))
    with pytest.raises(ProductionLipSyncError, match="exact rendered video"):
        _validate(report, video, audio)

    other_audio = tmp_path / "other.wav"
    other_audio.write_bytes(b"different-dialogue")
    report = _report(video, audio, dialogue_audio_sha256=_sha(other_audio))
    with pytest.raises(ProductionLipSyncError, match="exact approved dialogue audio"):
        _validate(report, video, audio)


def test_rejects_identity_or_cue_mismatch(tmp_path):
    video, audio = _artifacts(tmp_path)
    for field, value in [
        ("shot_id", "shot-99"),
        ("character_id", "char-b"),
        ("dialogue_cue_id", "cue-99"),
    ]:
        with pytest.raises(ProductionLipSyncError, match=field):
            _validate(_report(video, audio, **{field: value}), video, audio)


def test_rejects_unmeasured_or_unattributed_semantic_scores(tmp_path):
    video, audio = _artifacts(tmp_path)

    with pytest.raises(ProductionLipSyncError, match="scorer_name"):
        _validate(_report(video, audio, scorer_name=""), video, audio)
    with pytest.raises(ProductionLipSyncError, match="scorer_provenance"):
        _validate(_report(video, audio, scorer_provenance=""), video, audio)
    with pytest.raises(ProductionLipSyncError, match="measured frames and words"):
        _validate(_report(video, audio, measured_frame_count=0), video, audio)
    with pytest.raises(ProductionLipSyncError, match="measured frames and words"):
        _validate(_report(video, audio, measured_word_count=0), video, audio)


def test_returns_reject_decision_when_measured_quality_misses_policy(tmp_path):
    video, audio = _artifacts(tmp_path)
    evidence = _validate(
        _report(
            video,
            audio,
            sync_confidence=0.70,
            mean_offset_ms=145.0,
            p95_offset_ms=260.0,
        ),
        video,
        audio,
    )

    assert evidence.accepted is False
    assert evidence.decision == "reject"
    assert evidence.failed_metrics == (
        "sync_confidence",
        "mean_offset_ms",
        "p95_offset_ms",
    )


def test_custom_policy_is_validated_and_applied(tmp_path):
    video, audio = _artifacts(tmp_path)
    policy = ProductionLipSyncPolicy(
        minimum_sync_confidence=0.90,
        maximum_mean_offset_ms=60.0,
        maximum_p95_offset_ms=100.0,
    )
    evidence = _validate(_report(video, audio), video, audio, policy=policy)
    assert evidence.accepted is True

    with pytest.raises(ValueError, match="p95"):
        ProductionLipSyncPolicy(
            maximum_mean_offset_ms=200.0, maximum_p95_offset_ms=100.0
        )
