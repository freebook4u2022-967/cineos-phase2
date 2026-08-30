import pytest

from cineos.atlas.sequence_quality import (
    CineosSequenceQualityEvaluator,
    SequenceQualityError,
    SequenceQualityPolicy,
)


class Shot:
    shot_id = "shot-01"


def _evaluate(metrics, *, policy=None):
    evaluator = CineosSequenceQualityEvaluator(
        lambda *_args, **_kwargs: metrics,
        policy=policy,
    )
    return evaluator("candidate.mp4", shot=Shot(), attempt_index=0)


def test_accepts_shot_when_core_metrics_and_overall_score_pass():
    report = _evaluate(
        {
            "identity_similarity": 0.93,
            "temporal_consistency": 0.91,
            "artifact_integrity": 1.0,
            "motion_quality": 0.88,
            "anatomy_quality": 0.86,
        }
    )

    assert report["accepted"] is True
    assert report["decision"] == "accept"
    assert report["failed_metrics"] == []
    assert report["score"] >= report["policy"]["overall_floor"]


def test_hard_identity_floor_rejects_even_if_other_metrics_are_high():
    report = _evaluate(
        {
            "identity_similarity": 0.70,
            "temporal_consistency": 0.99,
            "artifact_integrity": 1.0,
            "motion_quality": 0.99,
        }
    )

    assert report["accepted"] is False
    assert "identity_similarity" in report["failed_metrics"]
    assert any("identity" in item for item in report["directives"])


def test_optional_difficult_case_metrics_contribute_to_overall_score():
    policy = SequenceQualityPolicy(overall_floor=0.90)
    base = {
        "identity_similarity": 0.94,
        "temporal_consistency": 0.94,
        "artifact_integrity": 0.98,
        "motion_quality": 0.94,
    }
    strong = _evaluate(
        {
            **base,
            "anatomy_quality": 0.98,
            "object_interaction_quality": 0.98,
            "dialogue_lip_sync": 0.98,
        },
        policy=policy,
    )
    weak = _evaluate(
        {
            **base,
            "anatomy_quality": 0.10,
            "object_interaction_quality": 0.10,
            "dialogue_lip_sync": 0.10,
        },
        policy=policy,
    )

    assert strong["score"] > weak["score"]
    assert strong["accepted"] is True
    assert weak["accepted"] is False
    assert "overall_score" in weak["failed_metrics"]


def test_missing_core_metric_fails_closed():
    with pytest.raises(SequenceQualityError, match="motion_quality"):
        _evaluate(
            {
                "identity_similarity": 0.9,
                "temporal_consistency": 0.9,
                "artifact_integrity": 1.0,
            }
        )


def test_metric_outside_normalized_range_fails_closed():
    with pytest.raises(SequenceQualityError, match="between 0 and 1"):
        _evaluate(
            {
                "identity_similarity": 1.2,
                "temporal_consistency": 0.9,
                "artifact_integrity": 1.0,
                "motion_quality": 0.9,
            }
        )


def test_policy_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="overall_floor"):
        SequenceQualityPolicy(overall_floor=1.1)
