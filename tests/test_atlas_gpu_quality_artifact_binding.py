from types import SimpleNamespace

import pytest

from cineos.atlas.gpu_connected_benchmark import (
    GPUConnectedBenchmarkError,
    _validated_quality_report,
)


def _request():
    return SimpleNamespace(
        scene_id="scene-01",
        shot_id="shot-01",
        content_hash="a" * 64,
    )


def _receipt(output_sha256: str):
    return SimpleNamespace(output_sha256=output_sha256)


def _production_report(artifact_sha256: str):
    return {
        "accepted": True,
        "score": 0.94,
        "production_measurement_evidence": True,
        "measurement": {
            "schema": "cineos-sequence-quality-measurement/0.1",
            "observer_id": "cineos-artifact-video-observer/0.1",
            "artifact_sha256": artifact_sha256,
        },
    }


def test_production_quality_report_accepts_exact_rendered_artifact_hash():
    artifact_hash = "b" * 64

    normalized = _validated_quality_report(
        _production_report(artifact_hash),
        request=_request(),
        receipt=_receipt(artifact_hash),
    )

    assert normalized["production_measurement_evidence"] is True
    assert normalized["measurement"]["artifact_sha256"] == artifact_hash


def test_production_quality_report_rejects_stale_or_substituted_artifact_hash():
    with pytest.raises(
        GPUConnectedBenchmarkError,
        match="production quality measurement does not match rendered artifact hash",
    ):
        _validated_quality_report(
            _production_report("c" * 64),
            request=_request(),
            receipt=_receipt("d" * 64),
        )


def test_production_quality_report_fails_closed_without_render_receipt_hash():
    with pytest.raises(
        GPUConnectedBenchmarkError,
        match="production quality measurement does not match rendered artifact hash",
    ):
        _validated_quality_report(
            _production_report("e" * 64),
            request=_request(),
            receipt=None,
        )


def test_generic_regression_quality_report_keeps_backward_compatibility():
    report = {
        "accepted": True,
        "score": 0.91,
        "metrics": {
            "identity_similarity": 0.92,
            "temporal_consistency": 0.90,
            "artifact_integrity": 1.0,
            "motion_quality": 0.85,
        },
    }

    normalized = _validated_quality_report(report, request=_request())

    assert normalized["accepted"] is True
    assert normalized["score"] == 0.91
