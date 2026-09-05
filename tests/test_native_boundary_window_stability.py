from __future__ import annotations

import pytest

from cineos.native_video.boundary_stability import (
    BoundaryWindowStabilityPolicy,
    evaluate_boundary_window_stability,
)


def _frame(value: int, size: int = 8) -> bytes:
    return bytes([value]) * size


def test_boundary_window_stability_accepts_stable_temporal_evidence() -> None:
    report = evaluate_boundary_window_stability(
        (_frame(80), _frame(82), _frame(84)),
        (_frame(86), _frame(88), _frame(90)),
    )

    assert report.decision == "accept"
    assert report.accepted is True
    assert report.sample_count_per_side == 3
    assert report.outgoing_peak_mad == pytest.approx(2.0)
    assert report.incoming_peak_mad == pytest.approx(2.0)


def test_boundary_window_stability_rejects_transient_flicker_on_both_sides() -> None:
    # Corresponding cross-boundary frames differ only slightly, so a boundary-only
    # pair metric can look healthy even though both windows contain the same flash.
    report = evaluate_boundary_window_stability(
        (_frame(80), _frame(180), _frame(82)),
        (_frame(88), _frame(188), _frame(90)),
    )

    assert report.decision == "reject"
    assert report.accepted is False
    assert report.outgoing_peak_mad == pytest.approx(100.0)
    assert report.incoming_peak_mad == pytest.approx(100.0)
    assert any("transient temporal instability" in item for item in report.directives)


def test_boundary_window_stability_warns_on_moderate_instability() -> None:
    policy = BoundaryWindowStabilityPolicy(warn_peak_mad=20.0, reject_peak_mad=60.0)
    report = evaluate_boundary_window_stability(
        (_frame(80), _frame(105), _frame(82)),
        (_frame(88), _frame(90), _frame(92)),
        policy,
    )

    assert report.decision == "warn"
    assert report.accepted is True
    assert report.outgoing_peak_mad == pytest.approx(25.0)
    assert any("transient flicker" in item for item in report.directives)


def test_boundary_window_stability_fails_closed_on_malformed_windows() -> None:
    with pytest.raises(ValueError, match="at least two frames"):
        evaluate_boundary_window_stability((_frame(80),), (_frame(88),))

    with pytest.raises(ValueError, match="same number of frames"):
        evaluate_boundary_window_stability(
            (_frame(80), _frame(82), _frame(84)),
            (_frame(88), _frame(90)),
        )

    with pytest.raises(ValueError, match="equal sizes"):
        evaluate_boundary_window_stability(
            (_frame(80), _frame(82)),
            (_frame(88, size=4), _frame(90, size=4)),
        )


def test_boundary_window_stability_policy_validates_threshold_order() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        BoundaryWindowStabilityPolicy(warn_peak_mad=-1.0)

    with pytest.raises(ValueError, match="cannot exceed"):
        BoundaryWindowStabilityPolicy(warn_peak_mad=50.0, reject_peak_mad=40.0)
