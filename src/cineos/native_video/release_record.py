"""Cryptographically bind a released CINEOS film to production evidence.

A successful render and a passing quality gate are not sufficient provenance for a
long-lived production system: the final media artifact must remain auditable against
the exact runtime/model policy, authored plan, and measured quality evidence that
approved it.  This module creates a deterministic, fail-closed release record for
that boundary.

The record deliberately contains hashes rather than copied mutable objects.  It can
therefore be stored beside a film, in a registry, or in a future distributed artifact
store without coupling those systems to CINEOS Python classes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifact_integrity import NativeArtifactProvenance, provenance_for, verify_provenance
from .runtime_manifest import ProductionRuntimeManifest

FINAL_FILM_RELEASE_RECORD_SCHEMA = "cineos-final-film-release/0.1"


class FinalFilmReleaseRecordError(RuntimeError):
    """Raised when release evidence is malformed or no longer verifies."""


def _stable_value(value: Any) -> Any:
    """Convert supported evidence into deterministic JSON-safe values.

    Production provenance must fail closed instead of silently stringifying unknown
    objects.  Dataclasses and objects exposing ``as_dict()`` are accepted because
    those are the contracts used by CINEOS plans and quality reports.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _stable_value(asdict(value))
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _stable_value(as_dict())
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_stable_value(item) for item in value]
    raise TypeError(f"unsupported release evidence type: {type(value)!r}")


def canonical_sha256(value: Any) -> str:
    """Return a deterministic SHA-256 digest for supported structured evidence."""
    payload = json.dumps(
        _stable_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class FinalFilmReleaseRecord:
    """Immutable content-addressed approval record for one final movie artifact."""

    artifact_sha256: str
    artifact_bytes: int
    runtime_manifest_sha256: str
    native_model_manifest_sha256: str
    final_gate_policy_fingerprint: str
    plan_sha256: str
    quality_report_sha256: str
    decision: str
    schema: str = FINAL_FILM_RELEASE_RECORD_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "artifact_sha256",
            "runtime_manifest_sha256",
            "plan_sha256",
            "quality_report_sha256",
        ):
            digest = getattr(self, field_name)
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if self.artifact_bytes <= 0:
            raise ValueError("artifact_bytes must be positive")
        if not self.native_model_manifest_sha256.strip():
            raise ValueError("native_model_manifest_sha256 must not be empty")
        if not self.final_gate_policy_fingerprint.strip():
            raise ValueError("final_gate_policy_fingerprint must not be empty")
        if self.decision not in {"accept", "warn"}:
            raise ValueError("only accepted final films can receive a release record")
        if self.schema != FINAL_FILM_RELEASE_RECORD_SCHEMA:
            raise ValueError("unsupported final-film release record schema")

    def snapshot(self) -> dict[str, Any]:
        """Return deterministic JSON-safe state suitable for durable storage."""
        return asdict(self)

    @classmethod
    def restore(cls, payload: Mapping[str, Any]) -> "FinalFilmReleaseRecord":
        """Restore a release record while rejecting unknown schemas/fields."""
        if payload.get("schema") != FINAL_FILM_RELEASE_RECORD_SCHEMA:
            raise FinalFilmReleaseRecordError("unsupported final-film release record schema")
        expected = {
            "artifact_sha256",
            "artifact_bytes",
            "runtime_manifest_sha256",
            "native_model_manifest_sha256",
            "final_gate_policy_fingerprint",
            "plan_sha256",
            "quality_report_sha256",
            "decision",
            "schema",
        }
        unknown = sorted(set(payload).difference(expected))
        missing = sorted(expected.difference(payload))
        if missing:
            raise FinalFilmReleaseRecordError(
                "final-film release record is missing: " + ", ".join(missing)
            )
        if unknown:
            raise FinalFilmReleaseRecordError(
                "final-film release record has unknown fields: " + ", ".join(unknown)
            )
        try:
            return cls(**dict(payload))
        except (TypeError, ValueError) as exc:
            raise FinalFilmReleaseRecordError(str(exc)) from exc


def build_release_record(
    movie_path: str | Path,
    *,
    runtime_manifest: ProductionRuntimeManifest,
    plan: Any,
    quality_report: Any,
) -> FinalFilmReleaseRecord:
    """Create a release record only for a measured accepted/warn final film."""
    decision = str(getattr(quality_report, "decision", ""))
    if decision not in {"accept", "warn"}:
        raise FinalFilmReleaseRecordError(
            "refusing release record for final film without accepted quality evidence"
        )
    provenance = provenance_for(movie_path)
    return FinalFilmReleaseRecord(
        artifact_sha256=provenance.sha256,
        artifact_bytes=provenance.byte_size,
        runtime_manifest_sha256=canonical_sha256(runtime_manifest.snapshot()),
        native_model_manifest_sha256=runtime_manifest.native_model_manifest_sha256,
        final_gate_policy_fingerprint=runtime_manifest.final_gate_policy_fingerprint,
        plan_sha256=canonical_sha256(plan),
        quality_report_sha256=canonical_sha256(quality_report),
        decision=decision,
    )


def verify_release_record(
    record: FinalFilmReleaseRecord,
    movie_path: str | Path,
    *,
    runtime_manifest: ProductionRuntimeManifest,
    plan: Any,
    quality_report: Any,
) -> NativeArtifactProvenance:
    """Fail closed if any released artifact or approval evidence has changed."""
    try:
        provenance = verify_provenance(
            movie_path,
            sha256=record.artifact_sha256,
            byte_size=record.artifact_bytes,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise FinalFilmReleaseRecordError(str(exc)) from exc

    checks = {
        "runtime_manifest_sha256": canonical_sha256(runtime_manifest.snapshot()),
        "native_model_manifest_sha256": runtime_manifest.native_model_manifest_sha256,
        "final_gate_policy_fingerprint": runtime_manifest.final_gate_policy_fingerprint,
        "plan_sha256": canonical_sha256(plan),
        "quality_report_sha256": canonical_sha256(quality_report),
        "decision": str(getattr(quality_report, "decision", "")),
    }
    mismatches = [
        field_name
        for field_name, actual in checks.items()
        if getattr(record, field_name) != actual
    ]
    if mismatches:
        raise FinalFilmReleaseRecordError(
            "final-film release evidence mismatch: " + ", ".join(mismatches)
        )
    if record.decision not in {"accept", "warn"}:
        raise FinalFilmReleaseRecordError("release record is not accepted")
    return provenance


__all__ = [
    "FINAL_FILM_RELEASE_RECORD_SCHEMA",
    "FinalFilmReleaseRecord",
    "FinalFilmReleaseRecordError",
    "build_release_record",
    "canonical_sha256",
    "verify_release_record",
]
