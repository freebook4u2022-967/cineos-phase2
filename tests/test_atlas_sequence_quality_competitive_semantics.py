from types import SimpleNamespace

import pytest

from cineos.atlas.sequence_quality import (
    CineosSequenceQualityEvaluator,
    SequenceQualityError,
)


_BASE_METRICS = {
    "identity_similarity": 0.95,
    "temporal_consistency": 0.95,
    "artifact_integrity": 0.98,
    "motion_quality": 0.95,
}


@pytest.mark.parametrize(
    ("challenge", "metric"),
    [
        ("multi_character_interaction", "multi_character_interaction_quality"),
        ("walking_running", "locomotion_quality"),
        ("fast_camera_movement", "camera_motion_quality"),
        ("lighting_changes", "lighting_transition_quality"),
        ("physics", "physics_plausibility"),
    ],
)
def test_competitive_challenge_requires_dedicated_measured_metric(challenge, metric):
    shot = SimpleNamespace(metadata={"competitive_challenges": [challenge]})
    evaluator = CineosSequenceQualityEvaluator(lambda *_args, **_kwargs: dict(_BASE_METRICS))

    with pytest.raises(SequenceQualityError, match=metric):
        evaluator("unused.mp4", shot=shot, attempt_index=0)


@pytest.mark.parametrize(
    ("challenge", "metric"),
    [
        ("multi_character_interaction", "multi_character_interaction_quality"),
        ("walking_running", "locomotion_quality"),
        ("fast_camera_movement", "camera_motion_quality"),
        ("lighting_changes", "lighting_transition_quality"),
        ("physics", "physics_plausibility"),
    ],
)
def test_competitive_challenge_rejects_low_dedicated_metric(challenge, metric):
    shot = SimpleNamespace(metadata={"competitive_challenges": [challenge]})
    metrics = {**_BASE_METRICS, metric: 0.10}
    evaluator = CineosSequenceQualityEvaluator(lambda *_args, **_kwargs: metrics)

    report = evaluator("unused.mp4", shot=shot, attempt_index=0)

    assert report["accepted"] is False
    assert metric in report["failed_metrics"]
    assert report["required_challenge_metrics"] == {metric: challenge}


@pytest.mark.parametrize(
    ("challenge", "metric"),
    [
        ("multi_character_interaction", "multi_character_interaction_quality"),
        ("walking_running", "locomotion_quality"),
        ("fast_camera_movement", "camera_motion_quality"),
        ("lighting_changes", "lighting_transition_quality"),
        ("physics", "physics_plausibility"),
    ],
)
def test_competitive_challenge_accepts_strong_dedicated_metric(challenge, metric):
    shot = SimpleNamespace(metadata={"competitive_challenges": [challenge]})
    metrics = {**_BASE_METRICS, metric: 0.95}
    evaluator = CineosSequenceQualityEvaluator(lambda *_args, **_kwargs: metrics)

    report = evaluator("unused.mp4", shot=shot, attempt_index=0)

    assert report["accepted"] is True
    assert report["required_challenge_metrics"] == {metric: challenge}


def test_generic_shot_remains_backward_compatible_without_specialist_metrics():
    shot = SimpleNamespace(metadata={})
    evaluator = CineosSequenceQualityEvaluator(lambda *_args, **_kwargs: dict(_BASE_METRICS))

    report = evaluator("unused.mp4", shot=shot, attempt_index=0)

    assert report["accepted"] is True
    assert report["required_challenge_metrics"] == {}
