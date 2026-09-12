import pytest

from cineos.atlas.sequence_quality import (
    CineosSequenceQualityEvaluator,
    SequenceQualityError,
)

BASE_METRICS = {
    "identity_similarity": 0.95,
    "temporal_consistency": 0.95,
    "artifact_integrity": 0.99,
    "motion_quality": 0.95,
}


class Shot:
    def __init__(self, metadata=None):
        self.metadata = metadata or {}


def _evaluate(metrics, *, metadata=None):
    evaluator = CineosSequenceQualityEvaluator(
        lambda *_args, **_kwargs: metrics,
    )
    return evaluator(
        "candidate.mp4",
        shot=Shot(metadata),
        attempt_index=0,
    )


def test_hands_challenge_requires_dedicated_anatomy_measurement():
    with pytest.raises(SequenceQualityError, match="anatomy_quality"):
        _evaluate(
            BASE_METRICS,
            metadata={"competitive_challenges": ["hands_anatomy"]},
        )


def test_low_anatomy_score_rejects_even_when_core_and_overall_are_strong():
    report = _evaluate(
        {**BASE_METRICS, "anatomy_quality": 0.20},
        metadata={"competitive_challenges": ["hands_anatomy"]},
    )

    assert report["accepted"] is False
    assert "anatomy_quality" in report["failed_metrics"]
    assert report["required_challenge_metrics"] == {"anatomy_quality": "hands_anatomy"}


def test_object_interaction_challenge_requires_dedicated_interaction_measurement():
    with pytest.raises(SequenceQualityError, match="object_interaction_quality"):
        _evaluate(
            BASE_METRICS,
            metadata={"competitive_challenges": ["object_interaction"]},
        )


def test_dialogue_challenge_alias_requires_measured_lip_sync():
    with pytest.raises(SequenceQualityError, match="dialogue_lip_sync"):
        _evaluate(
            BASE_METRICS,
            metadata={"benchmark_challenges": ["dialogue"]},
        )


def test_low_lip_sync_score_rejects_seedance_style_dialogue_shot():
    report = _evaluate(
        {**BASE_METRICS, "dialogue_lip_sync": 0.40},
        metadata={"benchmark_challenges": ["dialogue"]},
    )

    assert report["accepted"] is False
    assert "dialogue_lip_sync" in report["failed_metrics"]


def test_generic_shot_keeps_optional_metric_backward_compatibility():
    report = _evaluate({**BASE_METRICS, "anatomy_quality": 0.20})

    assert report["accepted"] is True
    assert "anatomy_quality" not in report["failed_metrics"]
    assert report["required_challenge_metrics"] == {}
