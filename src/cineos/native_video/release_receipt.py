"""Cryptographic production release receipts for native CINEOS films.

A successful render is not sufficient evidence that a movie is safe to publish.
Production needs a durable provenance record binding the exact final artifact to the
authored shot plan, runtime/model release, final measured QC evidence, and the final
``FilmBuild`` content hash. This module provides that fail-closed release boundary.

The receipt never generates or repairs media. It only hashes and verifies already
produced artifacts and JSON-safe production evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .runtime_manifest import (
    LEGACY_UNBOUND_NATIVE_MODEL_MANIFEST,
    ProductionRuntimeManifest,
)

PRODUCTION_FILM_RECEIPT_SCHEMA = "cineos-production-film-receipt/0.1"


class ProductionReleaseError(ValueError):
    """Raised when production release evidence is missing, invalid, or tampered."""


def _json_safe(value: Any) -> Any:
    """Return a deterministic JSON-safe representation or fail closed.

    Only explicit value types are accepted. Arbitrary objects are rejected instead
    of serializing ``repr``/``__dict__`` data whose ordering or contents may change
    across processes and silently weaken provenance checks.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    raise TypeError(
        "production release evidence must be composed of JSON-safe values or "
        "dataclasses; unsupported type: " + type(value).__qualname__
    )


def canonical_sha256(value: Any) -> str:
    """Hash deterministic JSON evidence with SHA-256."""
    encoded = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_sha256(path: str | Path) -> tuple[str, int]:
    """Return SHA-256 and byte size for a non-empty final movie artifact."""
    source = Path(path)
    if not source.is_file():
        raise ProductionReleaseError(f"final movie artifact does not exist: {source}")
    size = source.stat().st_size
    if size <= 0:
        raise ProductionReleaseError("final movie artifact is empty")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), size


def _qc_payload(report: Any) -> tuple[str, dict[str, Any]]:
    """Extract validated measured final-QC evidence."""
    raw_decision = (
        report.get("decision", "")
        if isinstance(report, Mapping)
        else getattr(report, "decision", "")
    )
    decision = str(raw_decision).strip().lower()
    if decision not in {"accept", "warn", "reject"}:
        raise ProductionReleaseError(
            "final QC report decision must be accept, warn, or reject"
        )
    as_dict = getattr(report, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
    elif is_dataclass(report) and not isinstance(report, type):
        payload = asdict(report)
    elif isinstance(report, Mapping):
        payload = dict(report)
    else:
        raise ProductionReleaseError(
            "final QC report must expose as_dict(), be a dataclass, or be a mapping"
        )
    if not isinstance(payload, Mapping):
        raise ProductionReleaseError("final QC evidence must be a mapping")
    safe = _json_safe(payload)
    payload_decision = str(safe.get("decision", decision)).strip().lower()
    if payload_decision != decision:
        raise ProductionReleaseError(
            "final QC decision does not match serialized QC evidence"
        )
    safe["decision"] = decision
    return decision, safe


@dataclass(frozen=True, slots=True)
class ProductionFilmReceipt:
    """Immutable provenance binding for one accepted production film artifact."""

    artifact_sha256: str
    artifact_size_bytes: int
    plan_sha256: str
    runtime_manifest_sha256: str
    final_qc_sha256: str
    final_qc_decision: str
    build_content_hash: str
    renderer_id: str
    native_model_manifest_sha256: str
    schema: str = PRODUCTION_FILM_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        for name in (
            "artifact_sha256",
            "plan_sha256",
            "runtime_manifest_sha256",
            "final_qc_sha256",
            "build_content_hash",
            "renderer_id",
            "native_model_manifest_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProductionReleaseError(f"{name} must be a non-empty string")
        if self.artifact_size_bytes <= 0:
            raise ProductionReleaseError("artifact_size_bytes must be positive")
        if self.final_qc_decision not in {"accept", "warn"}:
            raise ProductionReleaseError(
                "production film receipt requires accepted final QC"
            )
        if self.schema != PRODUCTION_FILM_RECEIPT_SCHEMA:
            raise ProductionReleaseError("unsupported production film receipt schema")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.as_dict())


def create_production_film_receipt(
    movie_path: str | Path,
    plan: Any,
    runtime_manifest: ProductionRuntimeManifest,
    final_qc_report: Any,
    *,
    build_content_hash: str,
    require_released_model: bool = True,
) -> ProductionFilmReceipt:
    """Create a release receipt only from accepted, fully bound production evidence."""
    if not isinstance(runtime_manifest, ProductionRuntimeManifest):
        raise TypeError("runtime_manifest must be a ProductionRuntimeManifest")
    if not isinstance(build_content_hash, str) or not build_content_hash.strip():
        raise ProductionReleaseError("build_content_hash must be a non-empty string")
    if (
        require_released_model
        and runtime_manifest.native_model_manifest_sha256
        == LEGACY_UNBOUND_NATIVE_MODEL_MANIFEST
    ):
        raise ProductionReleaseError(
            "production release requires a bound native model manifest"
        )

    decision, qc_payload = _qc_payload(final_qc_report)
    if decision == "reject":
        raise ProductionReleaseError(
            "rejected final QC cannot produce a release receipt"
        )

    movie_hash, movie_size = artifact_sha256(movie_path)
    runtime_payload = runtime_manifest.snapshot()
    return ProductionFilmReceipt(
        artifact_sha256=movie_hash,
        artifact_size_bytes=movie_size,
        plan_sha256=canonical_sha256(plan),
        runtime_manifest_sha256=canonical_sha256(runtime_payload),
        final_qc_sha256=canonical_sha256(qc_payload),
        final_qc_decision=decision,
        build_content_hash=build_content_hash,
        renderer_id=runtime_manifest.renderer_id,
        native_model_manifest_sha256=runtime_manifest.native_model_manifest_sha256,
    )


def verify_production_film_receipt(
    receipt: ProductionFilmReceipt,
    movie_path: str | Path,
    plan: Any,
    runtime_manifest: ProductionRuntimeManifest,
    final_qc_report: Any,
    *,
    build_content_hash: str,
) -> None:
    """Recompute every provenance binding and reject any mismatch."""
    if not isinstance(receipt, ProductionFilmReceipt):
        raise TypeError("receipt must be a ProductionFilmReceipt")
    expected = create_production_film_receipt(
        movie_path,
        plan,
        runtime_manifest,
        final_qc_report,
        build_content_hash=build_content_hash,
        require_released_model=(
            receipt.native_model_manifest_sha256 != LEGACY_UNBOUND_NATIVE_MODEL_MANIFEST
        ),
    )
    mismatches = [
        name
        for name in (
            "artifact_sha256",
            "artifact_size_bytes",
            "plan_sha256",
            "runtime_manifest_sha256",
            "final_qc_sha256",
            "final_qc_decision",
            "build_content_hash",
            "renderer_id",
            "native_model_manifest_sha256",
            "schema",
        )
        if getattr(receipt, name) != getattr(expected, name)
    ]
    if mismatches:
        raise ProductionReleaseError(
            "production film receipt verification failed: " + ", ".join(mismatches)
        )


def save_production_film_receipt(
    receipt: ProductionFilmReceipt, path: str | Path
) -> Path:
    """Atomically persist a receipt with an independent integrity hash."""
    if not isinstance(receipt, ProductionFilmReceipt):
        raise TypeError("receipt must be a ProductionFilmReceipt")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "receipt": receipt.as_dict(),
        "receipt_sha256": receipt.receipt_sha256,
    }
    encoded = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return target


def load_production_film_receipt(path: str | Path) -> ProductionFilmReceipt:
    """Load a persisted receipt and verify its independent integrity hash."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProductionReleaseError(
            f"cannot read production film receipt: {error}"
        ) from error
    if not isinstance(document, dict):
        raise ProductionReleaseError("production film receipt root must be an object")
    payload = document.get("receipt")
    recorded_hash = document.get("receipt_sha256")
    if not isinstance(payload, dict):
        raise ProductionReleaseError(
            "production film receipt is missing receipt payload"
        )
    if not isinstance(recorded_hash, str) or not recorded_hash:
        raise ProductionReleaseError(
            "production film receipt is missing integrity hash"
        )
    if canonical_sha256(payload) != recorded_hash:
        raise ProductionReleaseError("production film receipt integrity hash mismatch")
    try:
        return ProductionFilmReceipt(**payload)
    except (TypeError, ValueError) as error:
        raise ProductionReleaseError(
            f"invalid production film receipt: {error}"
        ) from error


__all__ = [
    "PRODUCTION_FILM_RECEIPT_SCHEMA",
    "ProductionFilmReceipt",
    "ProductionReleaseError",
    "artifact_sha256",
    "canonical_sha256",
    "create_production_film_receipt",
    "load_production_film_receipt",
    "save_production_film_receipt",
    "verify_production_film_receipt",
]
