import math

import pytest

from cineos.atlas.temporal_lattice import (
    TemporalFramePlanError,
    plan_temporal_frames,
    trim_generated_frames,
)


def test_legacy_foundation_preserves_requested_frame_count() -> None:
    plan = plan_temporal_frames(5.0, 24.0)

    assert plan.requested_frames == 120
    assert plan.generated_frames == 120
    assert plan.export_frames == 120
    assert plan.requires_trim is False


def test_wan_style_lattice_rounds_up_before_inference() -> None:
    plan = plan_temporal_frames(5.0, 24.0, temporal_compression_ratio=4)

    assert plan.requested_frames == 120
    assert plan.generated_frames == 121
    assert plan.export_frames == 120
    assert plan.requires_trim is True
    assert (plan.generated_frames - 1) % 4 == 0


def test_lattice_keeps_already_native_frame_count() -> None:
    plan = plan_temporal_frames(121 / 24, 24.0, temporal_compression_ratio=4)

    assert plan.requested_frames == 121
    assert plan.generated_frames == 121
    assert plan.requires_trim is False


def test_trim_requires_exact_foundation_frame_count() -> None:
    plan = plan_temporal_frames(5.0, 24.0, temporal_compression_ratio=4)

    with pytest.raises(TemporalFramePlanError, match="expected 121, got 120"):
        trim_generated_frames(list(range(120)), plan)

    assert trim_generated_frames(list(range(121)), plan) == list(range(120))


def test_generation_limit_fails_before_expensive_inference() -> None:
    with pytest.raises(TemporalFramePlanError, match="121 > 120"):
        plan_temporal_frames(
            5.0,
            24.0,
            temporal_compression_ratio=4,
            max_generation_frames=120,
        )


@pytest.mark.parametrize(
    ("duration", "fps", "ratio"),
    [
        (0.0, 24.0, 4),
        (math.inf, 24.0, 4),
        (5.0, 0.0, 4),
        (5.0, math.nan, 4),
        (5.0, 24.0, 0),
        (5.0, 24.0, True),
    ],
)
def test_invalid_plans_fail_closed(duration: float, fps: float, ratio: int) -> None:
    with pytest.raises(TemporalFramePlanError):
        plan_temporal_frames(duration, fps, temporal_compression_ratio=ratio)
