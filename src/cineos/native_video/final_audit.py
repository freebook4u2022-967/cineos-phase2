"""Durable, versioned audit records for native final-film acceptance.

A production film should never be considered releasable only because an in-memory
quality gate returned ``accept``.  This module binds the measured gate report to
the exact encoded movie bytes and persists that evidence atomically.  The record
is intentionally standard-library only so it can be verified by release tooling,
CI, recovery jobs, or a future studio service without importing a media backend.
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


class FinalFilmAuditError(RuntimeError):
    """Raised when persisted final-film evidence is incomplete or untrustworthy."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if self.movie_size_bytes <= 0:
            raise ValueError("movie_size_bytes must be positive")
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
        """Return a stable JSON-serializable representation."""
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "movie_sha256": self.movie_sha256,
            "movie_size_bytes": self.movie_size_bytes,
            "decision": self.decision,
            "model_fingerprint": self.model_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "report": dict(self.report),
        }

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
) -> FinalFilmAuditRecord:
    """Load schema-validated evidence and optionally verify its movie artifact."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalFilmAuditError(f"unable to read final-film audit: {source}") from exc
    if not isinstance(payload, dict):
        raise FinalFilmAuditError("final-film audit must contain a JSON object")
    try:
        report = payload["report"]
        if not isinstance(report, dict):
            raise TypeError("report must be a JSON object")
        decision = str(payload["decision"])
        if str(report.get("decision", "")) != decision:
            raise FinalFilmAuditError("audit decision disagrees with measured report")
        record = FinalFilmAuditRecord(
            schema_version=str(payload["schema_version"]),
            created_at=str(payload["created_at"]),
            movie_sha256=str(payload["movie_sha256"]),
            movie_size_bytes=int(payload["movie_size_bytes"]),
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


__all__ = [
    "FINAL_FILM_AUDIT_SCHEMA",
    "FinalFilmAuditError",
    "FinalFilmAuditRecord",
    "load_final_film_audit",
    "write_final_film_audit",
]
