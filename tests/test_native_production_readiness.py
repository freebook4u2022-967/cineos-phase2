from __future__ import annotations

import pytest

from cineos.native_video.production_readiness import (
    ProductionReadinessEvidence,
    evaluate_production_readiness,
)
from cineos.native_video.runtime_manifest import ProductionRuntimeManifest

TEST_MODEL_MANIFEST_SHA256 = "a" * 64


def _runtime(**overrides: object) -> ProductionRuntimeManifest:
    payload: dict[str, object] = {
        "renderer_id": "cineos-native-temporal",
        "temporal_model_fingerprint": "temporal-sha256",
        "device": "cuda",
        "max_recovery_attempts": 2,
        "require_final_film_evaluation": True,
        "require_audio": True,
        "final_gate_policy_fingerprint": "gate-sha256",
        "native_model_manifest_sha256": TEST_MODEL_MANIFEST_SHA256,
    }
    payload.update(overrides)
    return ProductionRuntimeManifest(**payload)  # type: ignore[arg-type]


def _evidence(**overrides: object) -> ProductionReadinessEvidence:
    payload: dict[str, object] = {
        "runtime_manifest": _runtime(),
        "native_model_trained": True,
        "native_model_benchmark_passed": True,
        "temporal_continuity_benchmark_passed": True,
        "character_identity_benchmark_passed": True,
        "audio_dialogue_gate_passed": True,
        "full_film_e2e_passed": True,
        "release_audit_passed": True,
        "external_gpu_training_required": False,
    }
    payload.update(overrides)
    return ProductionReadinessEvidence(**payload)  # type: ignore[arg-type]


def test_readiness_accepts_only_complete_production_evidence():
    report = evaluate_production_readiness(_evidence())

    assert report.ready is True
    assert report.blockers == ()
    report.require_ready()


def test_readiness_rejects_legacy_unbound_runtime():
    runtime = ProductionRuntimeManifest(
        renderer_id="cineos-native-temporal",
        temporal_model_fingerprint="temporal-sha256",
        device="cpu",
        max_recovery_attempts=1,
        require_final_film_evaluation=True,
        require_audio=True,
    )

    report = evaluate_production_readiness(_evidence(runtime_manifest=runtime))

    assert report.ready is False
    assert (
        "production runtime is not bound to a native model manifest" in report.blockers
    )
    assert "production runtime is not bound to a final-gate policy" in report.blockers


def test_readiness_reports_all_missing_validation_evidence():
    report = evaluate_production_readiness(
        _evidence(
            native_model_trained=False,
            native_model_benchmark_passed=False,
            temporal_continuity_benchmark_passed=False,
            character_identity_benchmark_passed=False,
            audio_dialogue_gate_passed=False,
            full_film_e2e_passed=False,
            release_audit_passed=False,
            external_gpu_training_required=True,
        )
    )

    assert report.ready is False
    assert len(report.blockers) == 8
    assert "native model training is not complete" in report.blockers
    assert "full-film end-to-end validation has not passed" in report.blockers
    assert "external GPU/model training dependency remains" in report.blockers


def test_readiness_rejects_disabled_production_acceptance_controls():
    runtime = _runtime(
        require_final_film_evaluation=False,
        require_audio=False,
    )

    report = evaluate_production_readiness(_evidence(runtime_manifest=runtime))

    assert report.ready is False
    assert "final film evaluation is disabled" in report.blockers
    assert "production audio acceptance is disabled" in report.blockers


def test_require_ready_raises_with_actionable_blockers():
    report = evaluate_production_readiness(
        _evidence(character_identity_benchmark_passed=False)
    )

    with pytest.raises(RuntimeError, match="character identity benchmark"):
        report.require_ready()


def test_readiness_rejects_wrong_evidence_type():
    with pytest.raises(TypeError, match="ProductionReadinessEvidence"):
        evaluate_production_readiness(object())  # type: ignore[arg-type]
