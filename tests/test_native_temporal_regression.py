from __future__ import annotations

import pytest

from cineos.native_video.temporal_regression import (
    TemporalRegressionPolicy,
    TemporalRegressionSnapshot,
    compare_temporal_regression,
)


def _snapshot(**overrides) -> TemporalRegressionSnapshot:
    values = {
        "benchmark_id": "cineos-release-film-v1",
        "frame_count": 120,
        "black_frame_ratio": 0.0,
        "frozen_transition_ratio": 0.04,
        "hard_cut_transition_ratio": 0.20,
        "mean_interframe_mad": 18.0,
        "scene_boundary_reject_count": 0,
        "scene_boundary_warn_count": 1,
        "mean_boundary_mad": 24.0,
    }
    values.update(overrides)
    return TemporalRegressionSnapshot(**values)


def test_temporal_regression_accepts_equal_or_improved_candidate() -> None:
    baseline = _snapshot()
    candidate = _snapshot(
        frozen_transition_ratio=0.02,
        hard_cut_transition_ratio=0.18,
        mean_interframe_mad=20.0,
        scene_boundary_warn_count=0,
    )

    report = compare_temporal_regression(baseline, candidate)

    assert report.decision == "accept"
    assert report.accepted is True
    assert report.directives == ()
    assert report.motion_retention == pytest.approx(20.0 / 18.0)


def test_temporal_regression_rejects_black_and_frozen_quality_loss() -> None:
    report = compare_temporal_regression(
        _snapshot(),
        _snapshot(black_frame_ratio=0.02, frozen_transition_ratio=0.10),
    )

    assert report.decision == "reject"
    assert report.accepted is False
    assert any("black-frame" in item for item in report.directives)
    assert any("frozen-transition" in item for item in report.directives)


def test_temporal_regression_rejects_new_boundary_failures() -> None:
    report = compare_temporal_regression(
        _snapshot(),
        _snapshot(scene_boundary_reject_count=1, scene_boundary_warn_count=2),
    )

    assert report.decision == "reject"
    assert report.boundary_reject_delta == 1
    assert report.boundary_warn_delta == 1
    assert any("reject count" in item for item in report.directives)
    assert any("warning count" in item for item in report.directives)


def test_temporal_regression_rejects_motion_collapse() -> None:
    report = compare_temporal_regression(
        _snapshot(mean_interframe_mad=20.0),
        _snapshot(mean_interframe_mad=8.0),
        TemporalRegressionPolicy(min_motion_retention=0.75),
    )

    assert report.decision == "reject"
    assert report.motion_retention == pytest.approx(0.4)
    assert any("motion collapsed" in item for item in report.directives)


def test_temporal_regression_fails_closed_when_benchmark_workload_changes() -> None:
    report = compare_temporal_regression(_snapshot(), _snapshot(frame_count=119))

    assert report.decision == "reject"
    assert any("frame count differs" in item for item in report.directives)


def test_temporal_regression_allows_explicit_versioned_frame_count_migration() -> None:
    report = compare_temporal_regression(
        _snapshot(),
        _snapshot(frame_count=119),
        TemporalRegressionPolicy(require_same_frame_count=False),
    )

    assert report.decision == "accept"


def test_temporal_regression_requires_same_benchmark_identity() -> None:
    with pytest.raises(ValueError, match="benchmark_id must match"):
        compare_temporal_regression(
            _snapshot(benchmark_id="film-a"),
            _snapshot(benchmark_id="film-b"),
        )


def test_temporal_regression_snapshot_validates_evidence_ranges() -> None:
    with pytest.raises(ValueError, match="temporal ratios"):
        _snapshot(black_frame_ratio=1.1)

    with pytest.raises(ValueError, match="frame_count"):
        _snapshot(frame_count=0)
