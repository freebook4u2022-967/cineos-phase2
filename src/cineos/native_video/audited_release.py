"""Fail-closed binding between a production release bundle and final-film audit.

The existing production release bundle proves that a film receipt, readiness
attestation, and runtime composition agree. Final-film audit evidence separately
proves that measured QC applies to the exact encoded movie bytes. A production V1
release must cross both trust boundaries together: operators should not be able to
reuse a valid bundle with an unrelated audit record, or a valid audit with a
release created under a different runtime/model composition.

This module adds that stricter boundary without changing the existing release-bundle
schema, preserving backwards compatibility for older tooling while giving new
production callers a fail-closed audited release contract.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .final_audit import (
    AUDIT_RECORD_SHA256_FIELD,
    FinalFilmAuditError,
    verify_production_final_film_audit_for_runtime,
)
from .release_bundle import ProductionReleaseBundle
from .release_receipt import ProductionReleaseError, canonical_sha256
from .runtime_manifest import ProductionRuntimeManifest

AUDITED_PRODUCTION_RELEASE_SCHEMA = "cineos-audited-production-release/0.1"


@dataclass(frozen=True, slots=True)
class AuditedProductionRelease:
    """Immutable release identity bound to final movie QC evidence."""

    release_bundle_sha256: str
    audit_record_sha256: str
    movie_sha256: str
    model_fingerprint: str
    runtime_fingerprint: str
    schema: str = AUDITED_PRODUCTION_RELEASE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != AUDITED_PRODUCTION_RELEASE_SCHEMA:
            raise ProductionReleaseError(
                "unsupported audited production release schema"
            )
        for field_name in (
            "release_bundle_sha256",
            "audit_record_sha256",
            "movie_sha256",
            "model_fingerprint",
            "runtime_fingerprint",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value.lower())
            ):
                raise ProductionReleaseError(
                    f"{field_name} must be one SHA-256 hex digest"
                )
            object.__setattr__(self, field_name, value.lower())

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """Canonical content identity of the complete audited release contract."""
        return canonical_sha256(self.as_dict())


def _audit_record_digest(record: Any) -> str:
    payload = record.to_record()
    digest = payload.get(AUDIT_RECORD_SHA256_FIELD)
    if not isinstance(digest, str):
        raise FinalFilmAuditError("final-film audit has no record integrity digest")
    return digest


def create_audited_production_release(
    bundle: ProductionReleaseBundle,
    audit_path: str | Path,
    movie_path: str | Path,
    runtime_manifest: ProductionRuntimeManifest,
    *,
    allow_warn: bool = False,
) -> AuditedProductionRelease:
    """Bind a verified final-film audit to one concrete production release bundle."""
    if not isinstance(bundle, ProductionReleaseBundle):
        raise TypeError("bundle must be ProductionReleaseBundle")
    if not isinstance(runtime_manifest, ProductionRuntimeManifest):
        raise TypeError("runtime_manifest must be a ProductionRuntimeManifest")
    if bundle.runtime_manifest_fingerprint != runtime_manifest.fingerprint:
        raise ProductionReleaseError(
            "release bundle runtime fingerprint does not match production runtime"
        )

    record = verify_production_final_film_audit_for_runtime(
        audit_path,
        movie_path=movie_path,
        runtime_manifest=runtime_manifest,
        allow_warn=allow_warn,
    )
    return AuditedProductionRelease(
        release_bundle_sha256=bundle.bundle_sha256,
        audit_record_sha256=_audit_record_digest(record),
        movie_sha256=record.movie_sha256,
        model_fingerprint=record.model_fingerprint,
        runtime_fingerprint=record.runtime_fingerprint,
    )


def verify_audited_production_release(
    release: AuditedProductionRelease,
    bundle: ProductionReleaseBundle,
    audit_path: str | Path,
    movie_path: str | Path,
    runtime_manifest: ProductionRuntimeManifest,
    *,
    allow_warn: bool = False,
) -> None:
    """Recompute the strict audited release binding and reject any provenance drift."""
    if not isinstance(release, AuditedProductionRelease):
        raise TypeError("release must be AuditedProductionRelease")
    expected = create_audited_production_release(
        bundle,
        audit_path,
        movie_path,
        runtime_manifest,
        allow_warn=allow_warn,
    )
    mismatches = [
        field_name
        for field_name in (
            "release_bundle_sha256",
            "audit_record_sha256",
            "movie_sha256",
            "model_fingerprint",
            "runtime_fingerprint",
            "schema",
        )
        if getattr(release, field_name) != getattr(expected, field_name)
    ]
    if mismatches:
        raise ProductionReleaseError(
            "audited production release verification failed: " + ", ".join(mismatches)
        )


def save_audited_production_release(
    release: AuditedProductionRelease, path: str | Path
) -> Path:
    """Atomically persist an audited release with envelope integrity protection."""
    if not isinstance(release, AuditedProductionRelease):
        raise TypeError("release must be AuditedProductionRelease")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "release": release.as_dict(),
        "release_sha256": release.fingerprint,
    }
    encoded = (
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
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
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_audited_production_release(path: str | Path) -> AuditedProductionRelease:
    """Load persisted audited release evidence and fail closed on tampering."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProductionReleaseError(
            f"cannot read audited production release: {error}"
        ) from error
    if not isinstance(document, dict):
        raise ProductionReleaseError(
            "audited production release root must be an object"
        )
    if set(document) != {"release", "release_sha256"}:
        raise ProductionReleaseError(
            "audited production release envelope fields invalid"
        )
    payload = document["release"]
    recorded_hash = document["release_sha256"]
    if not isinstance(payload, dict) or not isinstance(recorded_hash, str):
        raise ProductionReleaseError("audited production release envelope is malformed")
    if canonical_sha256(payload) != recorded_hash:
        raise ProductionReleaseError(
            "audited production release integrity hash mismatch"
        )
    try:
        release = AuditedProductionRelease(**payload)
    except (TypeError, ValueError) as error:
        raise ProductionReleaseError(
            f"invalid audited production release: {error}"
        ) from error
    if release.fingerprint != recorded_hash:
        raise ProductionReleaseError("audited production release fingerprint mismatch")
    return release


__all__ = [
    "AUDITED_PRODUCTION_RELEASE_SCHEMA",
    "AuditedProductionRelease",
    "create_audited_production_release",
    "load_audited_production_release",
    "save_audited_production_release",
    "verify_audited_production_release",
]
