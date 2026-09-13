import math

import pytest

from cineos.mission_one.dialogue import (
    LipSyncEvidence,
    LipSyncPolicy,
    evaluate_lip_sync,
    require_lip_sync_evidence,
    subtitle_entry,
)


def _evidence(**overrides) -> LipSyncEvidence:
    values = {
        "verifier_name": "reference-av-verifier",
        "verifier_version": "1.0",
        "verifier_origin": "external_pretrained_foundation",
        "model_id": "example/research-verifier",
        "model_license": "declared-by-runtime",
        "audio_sha256": "a" * 64,
        "video_sha256": "b" * 64,
        "offset_ms": 35.0,
        "confidence": 0.93,
        "speech_coverage": 0.82,
    }
    values.update(overrides)
    return LipSyncEvidence(**values)


def test_legacy_subtitle_entry_remains_unmeasured_by_default():
    entry = subtitle_entry("shot-1", "Hello", 1.0, 2.0)

    assert entry == {
        "shot_id": "shot-1",
        "text": "Hello",
        "start": 1.0,
        "end": 3.0,
        "lip_sync": "approximate_unless_measured",
    }


def test_passing_measurement_is_artifact_bound_and_provenance_explicit():
    evidence = _evidence()

    result = evaluate_lip_sync(evidence)

    assert result["passed"] is True
    assert result["status"] == "measured_pass"
    assert result["evidence"]["audio_sha256"] == "a" * 64
    assert result["evidence"]["video_sha256"] == "b" * 64
    assert result["evidence"]["verifier_origin"] == "external_pretrained_foundation"


def test_production_dialogue_fails_closed_without_measurement():
    with pytest.raises(ValueError, match="requires measured lip-sync evidence"):
        subtitle_entry(
            "shot-1",
            "Hello",
            0.0,
            1.0,
            require_measured_lip_sync=True,
        )


def test_production_dialogue_rejects_bad_av_offset():
    with pytest.raises(ValueError, match="av_offset_exceeds_limit"):
        require_lip_sync_evidence(_evidence(offset_ms=140.0))


def test_production_dialogue_rejects_low_confidence_or_coverage():
    result = evaluate_lip_sync(_evidence(confidence=0.4, speech_coverage=0.3))

    assert result["passed"] is False
    assert result["reasons"] == (
        "verifier_confidence_below_threshold",
        "speech_coverage_below_threshold",
    )


def test_measured_subtitle_records_pass_status_and_full_evidence():
    entry = subtitle_entry(
        "shot-2",
        "Measured line",
        0.5,
        1.5,
        lip_sync_evidence=_evidence(),
        require_measured_lip_sync=True,
    )

    assert entry["lip_sync"] == "measured_pass"
    assert entry["lip_sync_evidence"]["passed"] is True
    assert entry["lip_sync_evidence"]["schema"] == "cineos-lip-sync-evidence/0.1"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_measurements_are_rejected(value):
    with pytest.raises(ValueError):
        _evidence(offset_ms=value)


@pytest.mark.parametrize("field", ["audio_sha256", "video_sha256"])
def test_invalid_artifact_digest_is_rejected(field):
    with pytest.raises(ValueError, match=field):
        _evidence(**{field: "not-a-sha256"})


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError, match="max_abs_offset_ms"):
        LipSyncPolicy(max_abs_offset_ms=0.0)
    with pytest.raises(ValueError, match="min_confidence"):
        LipSyncPolicy(min_confidence=1.1)
