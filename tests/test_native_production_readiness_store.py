import hashlib
import json

import pytest

from cineos.native_video.production_readiness import (
    ProductionReadinessAttestation,
    ReadinessEvidenceArtifact,
)
from cineos.native_video.production_readiness_store import (
    PRODUCTION_READINESS_STORE_SCHEMA,
    ProductionReadinessStoreError,
    load_production_readiness_attestation,
    write_production_readiness_attestation,
)


def _attestation(tmp_path):
    evidence = tmp_path / "native-model-training.json"
    evidence.write_text('{"passed":true}', encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    return ProductionReadinessAttestation(
        runtime_manifest_fingerprint="a" * 64,
        artifacts=(
            ReadinessEvidenceArtifact(
                key="native_model_training",
                path=str(evidence),
                sha256=digest,
            ),
        ),
    )


def test_readiness_store_round_trip_is_stable(tmp_path):
    attestation = _attestation(tmp_path)
    destination = tmp_path / "readiness.json"

    written = write_production_readiness_attestation(attestation, destination)
    restored = load_production_readiness_attestation(
        written, expected_fingerprint=attestation.fingerprint
    )

    assert restored == attestation
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema"] == PRODUCTION_READINESS_STORE_SCHEMA
    assert payload["attestation_fingerprint"] == attestation.fingerprint


def test_readiness_store_rejects_tampered_attestation(tmp_path):
    attestation = _attestation(tmp_path)
    destination = write_production_readiness_attestation(
        attestation, tmp_path / "readiness.json"
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["attestation"]["runtime_manifest_fingerprint"] = "b" * 64
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProductionReadinessStoreError, match="fingerprint mismatch"):
        load_production_readiness_attestation(destination)


def test_readiness_store_rejects_unknown_schema(tmp_path):
    attestation = _attestation(tmp_path)
    destination = write_production_readiness_attestation(
        attestation, tmp_path / "readiness.json"
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["schema"] = "cineos-production-readiness-store/999"
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProductionReadinessStoreError, match="unsupported"):
        load_production_readiness_attestation(destination)


def test_readiness_store_rejects_untrusted_rollback(tmp_path):
    attestation = _attestation(tmp_path)
    destination = write_production_readiness_attestation(
        attestation, tmp_path / "readiness.json"
    )

    with pytest.raises(ProductionReadinessStoreError, match="trusted fingerprint"):
        load_production_readiness_attestation(
            destination, expected_fingerprint="f" * 64
        )


def test_readiness_store_rejects_unknown_fields(tmp_path):
    attestation = _attestation(tmp_path)
    destination = write_production_readiness_attestation(
        attestation, tmp_path / "readiness.json"
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["optimistic_override"] = True
    destination.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProductionReadinessStoreError, match="unknown fields"):
        load_production_readiness_attestation(destination)
