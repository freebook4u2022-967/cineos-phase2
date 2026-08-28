import hashlib
import json
from dataclasses import dataclass

import pytest

from cineos.native_video.production_readiness import (
    ProductionReadinessAttestation,
    ProductionReadinessEvidence,
    ReadinessEvidenceArtifact,
)
from cineos.native_video.release_bundle import (
    ProductionReleaseBundle,
    create_production_release_bundle,
    load_production_release_bundle,
    save_production_release_bundle,
    verify_production_release_bundle,
)
from cineos.native_video.release_receipt import (
    ProductionReleaseError,
    create_production_film_receipt,
)
from cineos.native_video.runtime_manifest import ProductionRuntimeManifest

TEST_MODEL_MANIFEST_SHA256 = "a" * 64


@dataclass(frozen=True)
class _FinalQC:
    decision: str = "accept"

    def as_dict(self):
        return {"decision": self.decision}


def _manifest():
    return ProductionRuntimeManifest(
        renderer_id="cineos-native",
        temporal_model_fingerprint="temporal-weights-sha",
        device="cpu",
        max_recovery_attempts=2,
        require_final_film_evaluation=True,
        require_audio=True,
        final_gate_policy_fingerprint="final-gate-policy-sha",
        native_model_manifest_sha256=TEST_MODEL_MANIFEST_SHA256,
    )


def _plan():
    return [{"shot_id": "shot-001", "scene_id": "scene-001", "duration": 4.0}]


def _readiness(tmp_path, manifest):
    artifacts = []
    for key in (
        "native_model_training",
        "native_model_benchmark",
        "temporal_continuity_benchmark",
        "character_identity_benchmark",
        "audio_dialogue_gate",
        "full_film_e2e",
        "release_audit",
    ):
        path = tmp_path / f"{key}.json"
        path.write_text('{"passed":true}', encoding="utf-8")
        artifacts.append(
            ReadinessEvidenceArtifact(
                key=key,
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    evidence = ProductionReadinessEvidence(
        runtime_manifest=manifest,
        native_model_trained=True,
        native_model_benchmark_passed=True,
        temporal_continuity_benchmark_passed=True,
        character_identity_benchmark_passed=True,
        audio_dialogue_gate_passed=True,
        full_film_e2e_passed=True,
        release_audit_passed=True,
    )
    attestation = ProductionReadinessAttestation(
        runtime_manifest_fingerprint=manifest.fingerprint,
        artifacts=tuple(artifacts),
    )
    return evidence, attestation


def _release_inputs(tmp_path):
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-final-movie")
    manifest = _manifest()
    qc = _FinalQC()
    receipt = create_production_film_receipt(
        movie,
        _plan(),
        manifest,
        qc,
        build_content_hash="film-build-content-sha",
    )
    evidence, attestation = _readiness(tmp_path, manifest)
    return movie, manifest, qc, receipt, evidence, attestation


def test_release_bundle_binds_receipt_readiness_and_runtime(tmp_path):
    movie, manifest, qc, receipt, evidence, attestation = _release_inputs(tmp_path)
    bundle = create_production_release_bundle(
        receipt,
        evidence,
        attestation,
        movie,
        _plan(),
        manifest,
        qc,
        build_content_hash="film-build-content-sha",
    )

    assert bundle.receipt_sha256 == receipt.receipt_sha256
    assert bundle.readiness_attestation_fingerprint == attestation.fingerprint
    assert bundle.runtime_manifest_fingerprint == manifest.fingerprint
    assert len(bundle.bundle_sha256) == 64
    verify_production_release_bundle(
        bundle,
        receipt,
        evidence,
        attestation,
        movie,
        _plan(),
        manifest,
        qc,
        build_content_hash="film-build-content-sha",
    )


def test_release_bundle_refuses_incomplete_readiness(tmp_path):
    movie, manifest, qc, receipt, evidence, attestation = _release_inputs(tmp_path)
    evidence = ProductionReadinessEvidence(
        runtime_manifest=manifest,
        native_model_trained=True,
        native_model_benchmark_passed=True,
        temporal_continuity_benchmark_passed=False,
        character_identity_benchmark_passed=True,
        audio_dialogue_gate_passed=True,
        full_film_e2e_passed=True,
        release_audit_passed=True,
    )

    with pytest.raises(ProductionReleaseError, match="temporal continuity benchmark"):
        create_production_release_bundle(
            receipt,
            evidence,
            attestation,
            movie,
            _plan(),
            manifest,
            qc,
            build_content_hash="film-build-content-sha",
        )


def test_release_bundle_refuses_tampered_readiness_artifact(tmp_path):
    movie, manifest, qc, receipt, evidence, attestation = _release_inputs(tmp_path)
    first = attestation.artifacts[0]
    with open(first.path, "w", encoding="utf-8") as handle:
        handle.write('{"passed":false}')

    with pytest.raises(ProductionReleaseError, match="digest mismatch"):
        create_production_release_bundle(
            receipt,
            evidence,
            attestation,
            movie,
            _plan(),
            manifest,
            qc,
            build_content_hash="film-build-content-sha",
        )


def test_release_bundle_detects_receipt_or_runtime_drift(tmp_path):
    movie, manifest, qc, receipt, evidence, attestation = _release_inputs(tmp_path)
    bundle = create_production_release_bundle(
        receipt,
        evidence,
        attestation,
        movie,
        _plan(),
        manifest,
        qc,
        build_content_hash="film-build-content-sha",
    )

    changed_plan = [{"shot_id": "shot-001", "scene_id": "scene-001", "duration": 5.0}]
    with pytest.raises(ProductionReleaseError, match="plan_sha256"):
        verify_production_release_bundle(
            bundle,
            receipt,
            evidence,
            attestation,
            movie,
            changed_plan,
            manifest,
            qc,
            build_content_hash="film-build-content-sha",
        )


def test_release_bundle_persistence_rejects_tampering(tmp_path):
    movie, manifest, qc, receipt, evidence, attestation = _release_inputs(tmp_path)
    bundle = create_production_release_bundle(
        receipt,
        evidence,
        attestation,
        movie,
        _plan(),
        manifest,
        qc,
        build_content_hash="film-build-content-sha",
    )
    path = save_production_release_bundle(bundle, tmp_path / "release-bundle.json")
    assert load_production_release_bundle(path) == bundle

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bundle"]["receipt_sha256"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProductionReleaseError, match="integrity hash mismatch"):
        load_production_release_bundle(path)


def test_release_bundle_rejects_invalid_digest_contract():
    with pytest.raises(ProductionReleaseError, match="receipt_sha256"):
        ProductionReleaseBundle(
            receipt_sha256="not-a-digest",
            readiness_attestation_fingerprint="a" * 64,
            runtime_manifest_fingerprint="b" * 64,
        )
