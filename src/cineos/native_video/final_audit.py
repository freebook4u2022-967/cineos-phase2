"""Durable, versioned audit records for native final-film acceptance.

A production film should never be considered releasable only because an in-memory
quality gate returned ``accept``. This module binds the measured gate report to
the exact encoded movie bytes and persists that evidence atomically. Audit payloads
also carry a canonical SHA-256 digest so post-write tampering with QC evidence is
detected independently from movie-artifact integrity.

The record is intentionally standard-library only so it can be verified by release
tooling, CI, recovery jobs, or a future studio service without importing a media
backend.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .final_gate import MeasuredFinalFilmReport

FINAL_FILM_AUDIT_SCHEMA = "cineos.native_video.final_film_audit.v1"
AUDIT_RECORD_SHA256_FIELD = "record_sha256"


class FinalFilmAuditError(RuntimeError):
    """Raised when persisted final-film evidence is incomplete or untrustworthy."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_record_sha256(payload: Mapping[str, Any]) -> str:
    """Return a stable digest for an audit payload excluding its own digest field."""
    material = {
        str(key): value
        for key, value in payload.items()
        if str(key) != AUDIT_RECORD_SHA256_FIELD
    }
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_record_digest(payload: Mapping[str, Any], *, required: bool) -> None:
    supplied = payload.get(AUDIT_RECORD_SHA256_FIELD)
    if supplied is None:
        if required:
            raise FinalFilmAuditError("final-film audit has no record integrity digest")
        return
    supplied_text = str(supplied)
    if len(supplied_text) != 64 or any(
        char not in "0123456789abcdef" for char in supplied_text
    ):
        raise FinalFilmAuditError("final-film audit record digest is malformed")
    expected = _canonical_record_sha256(payload)
    if supplied_text != expected:
        raise FinalFilmAuditError(
            "final-film audit record digest does not match payload"
        )


@dataclass(frozen=True, slots=True)
class FinalFilmAuditRecord:
    """Immutable acceptance evidence bound to one encoded movie artifact."""

    movie_sha256: str
    movie_size_bytes: int
    decision: str
    report: Mapping[str, Any]
    model_fingerprint: str = ""
    runtime_fingerprint: str = ""
    schema_version: str = FINAL_FILM_AUDIT_SCHEMA
    created_at: str = ""

    def __post_init__(self) -> None:
        if len(self.movie_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.movie_sha256
        ):
            raise ValueError("movie_sha256 must be a lowercase SHA-256 hex digest")
        if isinstance(self.movie_size_bytes, bool) or self.movie_size_bytes <= 0:
            raise ValueError("movie_size_bytes must be a positive integer")
        if self.decision not in {"accept", "warn", "reject"}:
            raise ValueError("decision must be accept, warn, or reject")
        if self.schema_version != FINAL_FILM_AUDIT_SCHEMA:
            raise ValueError("unsupported final-film audit schema")
        report_decision = str(self.report.get("decision", ""))
        if report_decision != self.decision:
            raise ValueError("audit decision must match measured report decision")
        if not self.created_at:
            object.__setattr__(self, "created_at", datetime.now(UTC).isoformat())

    @classmethod
    def from_report(
        cls,
        movie_path: str | Path,
        report: MeasuredFinalFilmReport,
        *,
        model_fingerprint: str = "",
        runtime_fingerprint: str = "",
    ) -> FinalFilmAuditRecord:
        """Bind a measured report to the exact bytes that were evaluated."""
        source = Path(movie_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        size = source.stat().st_size
        if size <= 0:
            raise FinalFilmAuditError("cannot audit an empty movie artifact")
        return cls(
            movie_sha256=_sha256_file(source),
            movie_size_bytes=size,
            decision=report.decision,
            report=report.as_dict(),
            model_fingerprint=model_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
        )

    def to_record(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation with self-integrity."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "movie_sha256": self.movie_sha256,
            "movie_size_bytes": self.movie_size_bytes,
            "decision": self.decision,
            "model_fingerprint": self.model_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "report": dict(self.report),
        }
        payload[AUDIT_RECORD_SHA256_FIELD] = _canonical_record_sha256(payload)
        return payload

    def verify_movie(self, movie_path: str | Path) -> None:
        """Fail closed when the audited movie has changed or disappeared."""
        source = Path(movie_path)
        if not source.is_file():
            raise FinalFilmAuditError(f"audited movie does not exist: {source}")
        if source.stat().st_size != self.movie_size_bytes:
            raise FinalFilmAuditError("audited movie size no longer matches record")
        if _sha256_file(source) != self.movie_sha256:
            raise FinalFilmAuditError("audited movie digest no longer matches record")


def write_final_film_audit(
    path: str | Path,
    record: FinalFilmAuditRecord,
    *,
    fsync: bool = True,
) -> Path:
    """Atomically persist one audit record, replacing stale evidence safely."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    payload = json.dumps(record.to_record(), sort_keys=True, separators=(",", ":"))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(temporary, target)
        if fsync:
            directory_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def load_final_film_audit(
    path: str | Path,
    *,
    movie_path: str | Path | None = None,
    require_record_digest: bool = False,
) -> FinalFilmAuditRecord:
    """Load schema-validated evidence and optionally verify its movie artifact.

    Newly written records always include a canonical payload digest. Legacy v1
    records without that field remain readable by default for backwards
    compatibility. Production release/recovery callers should pass
    ``require_record_digest=True`` to fail closed on legacy unsigned evidence.
    """
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalFilmAuditError(f"unable to read final-film audit: {source}") from exc
    if not isinstance(payload, dict):
        raise FinalFilmAuditError("final-film audit must contain a JSON object")

    _validate_record_digest(payload, required=require_record_digest)

    try:
        report = payload["report"]
        if not isinstance(report, dict):
            raise TypeError("report must be a JSON object")
        decision = str(payload["decision"])
        if str(report.get("decision", "")) != decision:
            raise FinalFilmAuditError("audit decision disagrees with measured report")
        raw_movie_size = payload["movie_size_bytes"]
        if isinstance(raw_movie_size, bool) or not isinstance(raw_movie_size, int):
            raise TypeError("movie_size_bytes must be an integer")
        record = FinalFilmAuditRecord(
            schema_version=str(payload["schema_version"]),
            created_at=str(payload["created_at"]),
            movie_sha256=str(payload["movie_sha256"]),
            movie_size_bytes=raw_movie_size,
            decision=decision,
            model_fingerprint=str(payload.get("model_fingerprint", "")),
            runtime_fingerprint=str(payload.get("runtime_fingerprint", "")),
            report=report,
        )
    except FinalFilmAuditError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise FinalFilmAuditError("invalid final-film audit record") from exc
    if movie_path is not None:
        record.verify_movie(movie_path)
    return record


def verify_production_final_film_audit(
    path: str | Path,
    *,
    movie_path: str | Path,
    expected_model_fingerprint: str,
    expected_runtime_fingerprint: str,
    allow_warn: bool = False,
) -> FinalFilmAuditRecord:
    """Verify one final film against the exact production release contract.

    This is the release boundary, not a migration helper. It always requires the
    audit payload integrity digest, re-hashes the encoded movie, requires explicit
    model/runtime bindings, and compares them to caller-held expected fingerprints.
    By default only a measured ``accept`` decision is releasable; an operator may
    deliberately permit ``warn`` through ``allow_warn=True`` without ever allowing
    ``reject``.
    """
    expected_model = expected_model_fingerprint.strip()
    expected_runtime = expected_runtime_fingerprint.strip()
    if not expected_model:
        raise ValueError("expected_model_fingerprint must not be empty")
    if not expected_runtime:
        raise ValueError("expected_runtime_fingerprint must not be empty")

    record = load_final_film_audit(
        path,
        movie_path=movie_path,
        require_record_digest=True,
    )
    if not record.model_fingerprint:
        raise FinalFilmAuditError("final-film audit has no model fingerprint binding")
    if not record.runtime_fingerprint:
        raise FinalFilmAuditError("final-film audit has no runtime fingerprint binding")
    if record.model_fingerprint != expected_model:
        raise FinalFilmAuditError(
            "final-film audit model fingerprint does not match production release"
        )
    if record.runtime_fingerprint != expected_runtime:
        raise FinalFilmAuditError(
            "final-film audit runtime fingerprint does not match production release"
        )
    accepted_decisions = {"accept", "warn"} if allow_warn else {"accept"}
    if record.decision not in accepted_decisions:
        raise FinalFilmAuditError(
            f"final-film audit decision is not releasable: {record.decision}"
        )
    return record


__all__ = [
    "AUDIT_RECORD_SHA256_FIELD",
    "FINAL_FILM_AUDIT_SCHEMA",
    "FinalFilmAuditError",
    "FinalFilmAuditRecord",
    "load_final_film_audit",
    "verify_production_final_film_audit",
    "write_final_film_audit",
]
