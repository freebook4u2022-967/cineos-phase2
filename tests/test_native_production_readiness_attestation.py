from __future__ import annotations

import hashlib

import pytest

from cineos.native_video import production_readiness as production_readiness_module
from cineos.native_video.production_readiness import (
    PRODUCTION_READINESS_ATTESTATION_SCHEMA,
    READINESS_EVIDENCE_KEYS,
    ProductionReadinessAttestation,
    ProductionReadinessEvidence,
    ReadinessEvidenceArtifact,
    evaluate_attested_production_readiness,
)
from cineos.native_video.runtime_manifest import ProductionRuntimeManifest


def _runtime() -> ProductionRuntimeManifest:
    return ProductionRuntimeManifest(
        renderer_id="cineos-native-temporal",
        temporal_model_fingerprint="temporal-sha256",
        device="cuda",
        max_recovery_attempts=2,
        require_final_film_evaluation=True,
        require_audio=True,
        final_gate_policy_fingerprint="gate-sha256",
        native_model_manifest_sha256="manifest-sha256",
    )


def _evidence(runtime: ProductionRuntimeManifest) -> ProductionReadinessEvidence:
    return ProductionReadinessEvidence(
        runtime_manifest=runtime,
        native_model_trained=True,
        native_model_benchmark_passed=True,
        temporal_continuity_benchmark_passed=True,
        character_identity_benchmark_passed=True,
        audio_dialogue_gate_passed=True,
        full_film_e2e_passed=True,
        release_audit_passed=True,
    )


def _attestation(tmp_path, runtime: ProductionRuntimeManifest):
    artifacts = []
    for key in READINESS_EVIDENCE_KEYS:
        path = tmp_path / f"{key}.json"
        payload = (key + "\n").encode("utf-8")
        path.write_bytes(payload)
        artifacts.append(
            ReadinessEvidenceArtifact(
                key=key,
                path=str(path),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    return ProductionReadinessAttestation(
        runtime_manifest_fingerprint=runtime.fingerprint,
        artifacts=tuple(artifacts),
    )


def test_attested_readiness_accepts_complete_untampered_evidence(tmp_path):
    runtime = _runtime()
    report = evaluate_attested_production_readiness(
        _evidence(runtime), _attestation(tmp_path, runtime)
    )

    assert report.ready is True
    assert report.blockers == ()


def test_attested_readiness_rejects_missing_artifact(tmp_path):
    runtime = _runtime()
    attestation = _attestation(tmp_path, runtime)
    missing = attestation.artifacts[0]
    (tmp_path / f"{missing.key}.json").unlink()

    report = evaluate_attested_production_readiness(_evidence(runtime), attestation)

    assert report.ready is False
    assert f"readiness evidence artifact is missing: {missing.key}" in report.blockers


def test_attested_readiness_rejects_tampered_artifact(tmp_path):
    runtime = _runtime()
    attestation = _attestation(tmp_path, runtime)
    artifact = attestation.artifacts[0]
    (tmp_path / f"{artifact.key}.json").write_text("tampered\n", encoding="utf-8")

    report = evaluate_attested_production_readiness(_evidence(runtime), attestation)

    assert report.ready is False
    assert (
        f"readiness evidence artifact digest mismatch: {artifact.key}"
        in report.blockers
    )


def test_attested_readiness_rejects_non_regular_artifact(tmp_path):
    runtime = _runtime()
    attestation = _attestation(tmp_path, runtime)
    artifact = attestation.artifacts[0]
    path = tmp_path / f"{artifact.key}.json"
    path.unlink()
    path.mkdir()

    report = evaluate_attested_production_readiness(_evidence(runtime), attestation)

    assert report.ready is False
    assert (
        f"readiness evidence artifact is not a regular file: {artifact.key}"
        in report.blockers
    )


def test_attested_readiness_rejects_symlink_artifact(tmp_path):
    runtime = _runtime()
    attestation = _attestation(tmp_path, runtime)
    artifact = attestation.artifacts[0]
    path = tmp_path / f"{artifact.key}.json"
    target = tmp_path / "replacement.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation is unavailable in this environment")

    report = evaluate_attested_production_readiness(_evidence(runtime), attestation)

    assert report.ready is False
    assert (
        f"readiness evidence artifact is not a regular file: {artifact.key}"
        in report.blockers
    )


def test_attested_readiness_rejects_path_replacement_before_open(tmp_path, monkeypatch):
    runtime = _runtime()
    original = _attestation(tmp_path, runtime)
    target = original.artifacts[0]
    path = tmp_path / f"{target.key}.json"
    replacement_payload = b"replacement evidence\n"
    replacement = tmp_path / "replacement-evidence.json"
    replacement.write_bytes(replacement_payload)

    replacement_artifact = ReadinessEvidenceArtifact(
        key=target.key,
        path=str(path),
        sha256=hashlib.sha256(replacement_payload).hexdigest(),
    )
    attestation = ProductionReadinessAttestation(
        runtime_manifest_fingerprint=runtime.fingerprint,
        artifacts=(replacement_artifact, *original.artifacts[1:]),
    )

    real_open = production_readiness_module.os.open
    replaced = False

    def replace_before_open(candidate, flags):
        nonlocal replaced
        if not replaced and candidate == path:
            replaced = True
            path.unlink()
            replacement.replace(path)
        return real_open(candidate, flags)

    monkeypatch.setattr(production_readiness_module.os, "open", replace_before_open)

    report = evaluate_attested_production_readiness(_evidence(runtime), attestation)

    assert replaced is True
    assert report.ready is False
    assert (
        f"readiness evidence artifact changed during verification: {target.key}"
        in report.blockers
    )


def test_attested_readiness_rejects_runtime_mismatch(tmp_path):
    runtime = _runtime()
    attestation = _attestation(tmp_path, runtime)
    other_runtime = ProductionRuntimeManifest(
        renderer_id=runtime.renderer_id,
        temporal_model_fingerprint=runtime.temporal_model_fingerprint,
        device="cpu",
        max_recovery_attempts=runtime.max_recovery_attempts,
        require_final_film_evaluation=True,
        require_audio=True,
        final_gate_policy_fingerprint=runtime.final_gate_policy_fingerprint,
        native_model_manifest_sha256=runtime.native_model_manifest_sha256,
    )

    report = evaluate_attested_production_readiness(
        _evidence(other_runtime), attestation
    )

    assert report.ready is False
    assert (
        "readiness attestation is bound to a different runtime manifest"
        in report.blockers
    )


def test_attestation_rejects_duplicate_evidence_keys(tmp_path):
    runtime = _runtime()
    artifact = _attestation(tmp_path, runtime).artifacts[0]

    with pytest.raises(ValueError, match="duplicate readiness evidence keys"):
        ProductionReadinessAttestation(
            runtime_manifest_fingerprint=runtime.fingerprint,
            artifacts=(artifact, artifact),
        )


def test_attestation_snapshot_round_trip_is_stable(tmp_path):
    runtime = _runtime()
    attestation = _attestation(tmp_path, runtime)

    restored = ProductionReadinessAttestation.restore(attestation.snapshot())

    assert restored == attestation
    assert restored.schema == PRODUCTION_READINESS_ATTESTATION_SCHEMA
    assert restored.fingerprint == attestation.fingerprint


def test_attestation_restore_rejects_unknown_schema(tmp_path):
    runtime = _runtime()
    payload = _attestation(tmp_path, runtime).snapshot()
    payload["schema"] = "cineos-production-readiness-attestation/999"

    with pytest.raises(ValueError, match="unsupported production readiness"):
        ProductionReadinessAttestation.restore(payload)


def test_attestation_restore_rejects_unknown_contract_fields(tmp_path):
    runtime = _runtime()
    payload = _attestation(tmp_path, runtime).snapshot()
    payload["optimistic_override"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        ProductionReadinessAttestation.restore(payload)


def test_artifact_restore_rejects_unknown_contract_fields(tmp_path):
    runtime = _runtime()
    artifact = _attestation(tmp_path, runtime).artifacts[0]
    payload = artifact.snapshot()
    payload["passed"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        ReadinessEvidenceArtifact.restore(payload)
