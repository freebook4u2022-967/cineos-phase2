"""Production release bundle binding film provenance to readiness evidence.

A film receipt proves what was rendered, but production release must also prove that
CINEOS itself was ready to ship that artifact: the native model, temporal/identity
benchmarks, audio gate, full-film E2E, and release audit all need durable evidence.
This module joins those two trust boundaries without changing the existing receipt
schema, preserving backwards compatibility while providing a stricter V1 release
contract.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .production_readiness import (
    ProductionReadinessAttestation,
    ProductionReadinessEvidence,
    evaluate_attested_production_readiness,
)
from .release_receipt import (
    ProductionFilmReceipt,
    ProductionReleaseError,
    canonical_sha256,
    verify_production_film_receipt,
)
from .runtime_manifest import ProductionRuntimeManifest

PRODUCTION_RELEASE_BUNDLE_SCHEMA = "cineos-production-release-bundle/0.1"


@dataclass(frozen=True, slots=True)
class ProductionReleaseBundle:
    """Immutable binding between a film receipt and production-readiness proof."""

    receipt_sha256: str
    readiness_attestation_fingerprint: str
    runtime_manifest_fingerprint: str
    schema: str = PRODUCTION_RELEASE_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PRODUCTION_RELEASE_BUNDLE_SCHEMA:
            raise ProductionReleaseError("unsupported production release bundle schema")
        for name in (
            "receipt_sha256",
            "readiness_attestation_fingerprint",
            "runtime_manifest_fingerprint",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value.lower())
            ):
                raise ProductionReleaseError(f"{name} must be one SHA-256 hex digest")
            object.__setattr__(self, name, value.lower())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def bundle_sha256(self) -> str:
        return canonical_sha256(self.as_dict())


def create_production_release_bundle(
    receipt: ProductionFilmReceipt,
    readiness_evidence: ProductionReadinessEvidence,
    readiness_attestation: ProductionReadinessAttestation,
    movie_path: str | Path,
    plan: Any,
    runtime_manifest: ProductionRuntimeManifest,
    final_qc_report: Any,
    *,
    build_content_hash: str,
) -> ProductionReleaseBundle:
    """Create a bundle only when film provenance and readiness both verify."""
    readiness = evaluate_attested_production_readiness(
        readiness_evidence, readiness_attestation
    )
    try:
        readiness.require_ready()
    except RuntimeError as error:
        raise ProductionReleaseError(str(error)) from error

    if readiness_evidence.runtime_manifest != runtime_manifest:
        raise ProductionReleaseError(
            "readiness evidence runtime does not match release runtime manifest"
        )

    verify_production_film_receipt(
        receipt,
        movie_path,
        plan,
        runtime_manifest,
        final_qc_report,
        build_content_hash=build_content_hash,
    )

    return ProductionReleaseBundle(
        receipt_sha256=receipt.receipt_sha256,
        readiness_attestation_fingerprint=readiness_attestation.fingerprint,
        runtime_manifest_fingerprint=runtime_manifest.fingerprint,
    )


def verify_production_release_bundle(
    bundle: ProductionReleaseBundle,
    receipt: ProductionFilmReceipt,
    readiness_evidence: ProductionReadinessEvidence,
    readiness_attestation: ProductionReadinessAttestation,
    movie_path: str | Path,
    plan: Any,
    runtime_manifest: ProductionRuntimeManifest,
    final_qc_report: Any,
    *,
    build_content_hash: str,
) -> None:
    """Recompute the strict production release binding and reject any drift."""
    if not isinstance(bundle, ProductionReleaseBundle):
        raise TypeError("bundle must be ProductionReleaseBundle")
    expected = create_production_release_bundle(
        receipt,
        readiness_evidence,
        readiness_attestation,
        movie_path,
        plan,
        runtime_manifest,
        final_qc_report,
        build_content_hash=build_content_hash,
    )
    mismatches = [
        name
        for name in (
            "receipt_sha256",
            "readiness_attestation_fingerprint",
            "runtime_manifest_fingerprint",
            "schema",
        )
        if getattr(bundle, name) != getattr(expected, name)
    ]
    if mismatches:
        raise ProductionReleaseError(
            "production release bundle verification failed: " + ", ".join(mismatches)
        )


def save_production_release_bundle(
    bundle: ProductionReleaseBundle, path: str | Path
) -> Path:
    """Atomically persist a bundle with an independent content hash."""
    if not isinstance(bundle, ProductionReleaseBundle):
        raise TypeError("bundle must be ProductionReleaseBundle")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bundle": bundle.as_dict(),
        "bundle_sha256": bundle.bundle_sha256,
    }
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        try:
            directory_fd = os.open(
                destination.parent,
                getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY,
            )
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return destination


def load_production_release_bundle(path: str | Path) -> ProductionReleaseBundle:
    """Load a persisted bundle and fail closed on corruption or schema drift."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProductionReleaseError(
            f"cannot read production release bundle: {error}"
        ) from error
    if not isinstance(document, dict):
        raise ProductionReleaseError("production release bundle root must be an object")
    if set(document) != {"bundle", "bundle_sha256"}:
        raise ProductionReleaseError(
            "production release bundle envelope fields invalid"
        )
    payload = document["bundle"]
    recorded_hash = document["bundle_sha256"]
    if not isinstance(payload, dict) or not isinstance(recorded_hash, str):
        raise ProductionReleaseError("production release bundle envelope is malformed")
    if canonical_sha256(payload) != recorded_hash:
        raise ProductionReleaseError(
            "production release bundle integrity hash mismatch"
        )
    try:
        return ProductionReleaseBundle(**payload)
    except (TypeError, ValueError) as error:
        raise ProductionReleaseError(
            f"invalid production release bundle: {error}"
        ) from error


__all__ = [
    "PRODUCTION_RELEASE_BUNDLE_SCHEMA",
    "ProductionReleaseBundle",
    "create_production_release_bundle",
    "load_production_release_bundle",
    "save_production_release_bundle",
    "verify_production_release_bundle",
]
