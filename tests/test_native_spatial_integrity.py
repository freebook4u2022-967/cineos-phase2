from __future__ import annotations

import pytest

from cineos.native_video.spatial_integrity import (
    SpatialIntegrityPolicy,
    evaluate_spatial_samples,
)


def _flat(value: int, width: int = 4, height: int = 4) -> bytes:
    return bytes([value]) * (width * height)


def _checker(low: int, high: int, width: int = 4, height: int = 4) -> bytes:
    values = []
    for row in range(height):
        for column in range(width):
            values.append(high if (row + column) % 2 else low)
    return bytes(values)


def test_spatial_integrity_accepts_structured_picture_evidence() -> None:
    report = evaluate_spatial_samples(
        (_checker(30, 180), _checker(40, 190)),
        width=4,
        height=4,
    )

    assert report.decision == "accept"
    assert report.accepted is True
    assert report.low_detail_frame_ratio == 0.0
    assert report.mean_variance > 1000.0
    assert report.mean_edge_mad > 100.0


def test_spatial_integrity_rejects_non_black_featureless_decoder_output() -> None:
    report = evaluate_spatial_samples(
        (_flat(80), _flat(100), _flat(120)),
        width=4,
        height=4,
        policy=SpatialIntegrityPolicy(max_low_detail_ratio=0.20),
    )

    assert report.decision == "reject"
    assert report.accepted is False
    assert report.low_detail_frame_ratio == 1.0
    assert any("spatial structure" in item for item in report.directives)


def test_spatial_integrity_warns_on_isolated_low_detail_frame() -> None:
    report = evaluate_spatial_samples(
        (_checker(20, 180), _flat(110), _checker(30, 190), _checker(40, 200)),
        width=4,
        height=4,
        policy=SpatialIntegrityPolicy(max_low_detail_ratio=0.50),
    )

    assert report.decision == "warn"
    assert report.accepted is True
    assert report.low_detail_frame_ratio == pytest.approx(0.25)
    assert any("renderer collapse" in item for item in report.directives)


def test_spatial_integrity_requires_declared_frame_geometry() -> None:
    with pytest.raises(ValueError, match="declared dimensions"):
        evaluate_spatial_samples((b"abc",), width=2, height=2)


def test_spatial_integrity_policy_validates_thresholds() -> None:
    with pytest.raises(ValueError, match="max_low_detail_ratio"):
        SpatialIntegrityPolicy(max_low_detail_ratio=1.1)
