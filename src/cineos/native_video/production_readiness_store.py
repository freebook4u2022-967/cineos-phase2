"""Crash-safe durable storage for CINEOS production-readiness attestations.

Production readiness is a release boundary, so an attestation must survive process
restarts without silently accepting a torn write or modified payload. This module
persists the versioned attestation in a small content-addressed envelope using an
atomic same-filesystem replacement and verifies the embedded fingerprint on load.

The referenced readiness evidence artifacts remain independently content-addressed
and are re-verified by ``evaluate_attested_production_readiness``. The store protects
the attestation contract itself; it does not convert missing training or benchmark
evidence into a passing result.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .production_readiness import ProductionReadinessAttestation

PRODUCTION_READINESS_STORE_SCHEMA = "cineos-production-readiness-store/0.1"


class ProductionReadinessStoreError(ValueError):
    """Raised when durable production-readiness state is invalid or unreadable."""


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _envelope(attestation: ProductionReadinessAttestation) -> dict[str, Any]:
    return {
        "schema": PRODUCTION_READINESS_STORE_SCHEMA,
        "attestation_fingerprint": attestation.fingerprint,
        "attestation": attestation.snapshot(),
    }


def write_production_readiness_attestation(
    attestation: ProductionReadinessAttestation,
    path: str | Path,
) -> Path:
    """Atomically persist one immutable readiness attestation envelope."""
    if not isinstance(attestation, ProductionReadinessAttestation):
        raise TypeError("attestation must be ProductionReadinessAttestation")

    destination = Path(path)
    payload = json.dumps(
        _envelope(attestation),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    try:
        _atomic_write_bytes(destination, payload)
    except OSError as error:
        raise ProductionReadinessStoreError(
            f"cannot persist production readiness attestation: {error}"
        ) from error
    return destination


def load_production_readiness_attestation(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
) -> ProductionReadinessAttestation:
    """Load durable readiness state and fail closed on corruption or rollback.

    ``expected_fingerprint`` is an optional caller-held trust anchor. When supplied,
    a valid but older attestation cannot be substituted without detection.
    """
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ProductionReadinessStoreError(
            "production readiness attestation file is missing"
        ) from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionReadinessStoreError(
            f"cannot load production readiness attestation: {error}"
        ) from error

    if not isinstance(payload, dict):
        raise ProductionReadinessStoreError(
            "production readiness store payload must be an object"
        )
    required = {"schema", "attestation_fingerprint", "attestation"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ProductionReadinessStoreError(
            "production readiness store is missing: " + ", ".join(missing)
        )
    unknown = sorted(set(payload).difference(required))
    if unknown:
        raise ProductionReadinessStoreError(
            "production readiness store has unknown fields: " + ", ".join(unknown)
        )
    if payload["schema"] != PRODUCTION_READINESS_STORE_SCHEMA:
        raise ProductionReadinessStoreError(
            "unsupported production readiness store schema"
        )

    attestation_payload = payload["attestation"]
    stored_fingerprint = payload["attestation_fingerprint"]
    if not isinstance(attestation_payload, dict):
        raise ProductionReadinessStoreError(
            "production readiness attestation payload must be an object"
        )
    if not isinstance(stored_fingerprint, str):
        raise ProductionReadinessStoreError(
            "production readiness attestation fingerprint must be a string"
        )

    try:
        attestation = ProductionReadinessAttestation.restore(attestation_payload)
    except (TypeError, ValueError) as error:
        raise ProductionReadinessStoreError(
            f"invalid production readiness attestation: {error}"
        ) from error

    if stored_fingerprint != attestation.fingerprint:
        raise ProductionReadinessStoreError(
            "production readiness attestation fingerprint mismatch"
        )
    if expected_fingerprint is not None:
        trusted = expected_fingerprint.strip().lower()
        if trusted != attestation.fingerprint:
            raise ProductionReadinessStoreError(
                "production readiness attestation differs from trusted fingerprint"
            )
    return attestation


__all__ = [
    "PRODUCTION_READINESS_STORE_SCHEMA",
    "ProductionReadinessStoreError",
    "load_production_readiness_attestation",
    "write_production_readiness_attestation",
]
