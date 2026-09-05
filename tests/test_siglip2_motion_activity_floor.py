"""Regression tests for conservative SigLIP2 motion activity evidence."""

import math

import pytest

from cineos.atlas.siglip2_video_scorer import (
    DEFAULT_MOTION_ACTIVITY_FLOOR,
    DEFAULT_MOTION_STEP_CEILING,
    DEFAULT_MOTION_SUPPORT_FRACTION,
    SIGLIP2_QC_SCHEMA,
    SigLIP2FeatureVideoScorer,
)


def _unit(angle: float) -> tuple[float, float]:
    return (math.cos(angle), math.sin(angle))


def test_two_frame_numerical_jitter_is_not_motion_evidence() -> None:
    score = SigLIP2FeatureVideoScorer._motion_coherence([_unit(0.0), _unit(0.001)])

    assert score == 0.0


def test_multi_frame_tiny_jitter_is_not_motion_evidence() -> None:
    score = SigLIP2FeatureVideoScorer._motion_coherence(
        [_unit(0.0), _unit(0.001), _unit(-0.001), _unit(0.0005)]
    )

    assert score == 0.0


def test_nontrivial_two_frame_change_can_supply_motion_evidence() -> None:
    score = SigLIP2FeatureVideoScorer._motion_coherence([_unit(0.0), _unit(0.1)])

    assert score == 1.0


def test_extreme_two_frame_jump_is_rejected_as_cut_like() -> None:
    score = SigLIP2FeatureVideoScorer._motion_coherence([_unit(0.0), _unit(math.pi)])

    assert score == 0.0


def test_extreme_internal_jump_invalidates_motion_evidence() -> None:
    score = SigLIP2FeatureVideoScorer._motion_coherence(
        [_unit(0.0), _unit(0.1), _unit(math.pi), _unit(math.pi + 0.1)]
    )

    assert score == 0.0


def test_stable_nontrivial_motion_remains_high_quality() -> None:
    score = SigLIP2FeatureVideoScorer._motion_coherence(
        [_unit(0.0), _unit(0.1), _unit(0.2), _unit(0.3)]
    )

    assert score > 0.99


def test_single_feature_jump_cannot_masquerade_as_sustained_motion() -> None:
    score = SigLIP2FeatureVideoScorer._motion_coherence(
        [_unit(0.0), _unit(0.3), _unit(0.3), _unit(0.3), _unit(0.3)]
    )

    assert score <= 0.25


def test_sparse_motion_below_required_support_fails_closed() -> None:
    score = SigLIP2FeatureVideoScorer._motion_coherence(
        [_unit(0.0), _unit(0.3), _unit(0.3), _unit(0.3), _unit(0.3), _unit(0.3)],
        support_fraction=0.5,
    )

    assert score == 0.0


def test_partial_motion_is_capped_by_temporal_support() -> None:
    score = SigLIP2FeatureVideoScorer._motion_coherence(
        [_unit(0.0), _unit(0.1), _unit(0.2), _unit(0.2), _unit(0.2)]
    )

    assert 0.0 < score <= 0.5


@pytest.mark.parametrize("floor", [0.0, -1.0, math.inf, math.nan, 1.01])
def test_motion_activity_floor_rejects_invalid_values(floor: float) -> None:
    with pytest.raises(ValueError, match="activity_floor"):
        SigLIP2FeatureVideoScorer._motion_coherence(
            [_unit(0.0), _unit(0.1)],
            activity_floor=floor,
        )


@pytest.mark.parametrize("fraction", [0.0, -1.0, math.inf, math.nan, 1.01])
def test_motion_support_fraction_rejects_invalid_values(fraction: float) -> None:
    with pytest.raises(ValueError, match="support_fraction"):
        SigLIP2FeatureVideoScorer._motion_coherence(
            [_unit(0.0), _unit(0.1)],
            support_fraction=fraction,
        )


@pytest.mark.parametrize("ceiling", [0.0, -1.0, math.inf, math.nan, 1.01])
def test_motion_step_ceiling_rejects_invalid_values(ceiling: float) -> None:
    with pytest.raises(ValueError, match="step_ceiling"):
        SigLIP2FeatureVideoScorer._motion_coherence(
            [_unit(0.0), _unit(0.1)],
            step_ceiling=ceiling,
        )


def test_motion_step_ceiling_must_exceed_activity_floor() -> None:
    with pytest.raises(ValueError, match="greater than activity_floor"):
        SigLIP2FeatureVideoScorer._motion_coherence(
            [_unit(0.0), _unit(0.1)],
            activity_floor=0.2,
            step_ceiling=0.2,
        )


def test_default_motion_policy_is_conservative_and_versioned() -> None:
    assert DEFAULT_MOTION_ACTIVITY_FLOOR == pytest.approx(1e-4)
    assert DEFAULT_MOTION_SUPPORT_FRACTION == pytest.approx(0.25)
    assert DEFAULT_MOTION_STEP_CEILING == pytest.approx(0.5)
    assert SIGLIP2_QC_SCHEMA == "cineos-external-siglip2-video-qc/0.7"
