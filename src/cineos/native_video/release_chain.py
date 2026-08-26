"""Tamper-evident production release lineage for native CINEOS films.

Per-film receipts prove one artifact's provenance. Production also needs a durable
history across renderer/model upgrades so an operator cannot silently replace or
roll back an accepted release. This module chains release receipts together using
canonical SHA-256 hashes and fails closed on malformed or broken lineage.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RELEASE_CHAIN_SCHEMA = "cineos-release-chain/0.1"
GENESIS_PREVIOUS_SHA256 = "0" * 64


class ReleaseChainError(ValueError):
    """Raised when release lineage is malformed, inconsistent, or tampered."""


def _require_sha256(value: str, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ReleaseChainError(f"{name} must be a 64-character SHA-256 hex digest")
    return normalized


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ReleaseChainEntry:
    """One immutable production release linked to its predecessor."""

    release_id: str
    receipt_sha256: str
    native_model_manifest_sha256: str
    previous_entry_sha256: str = GENESIS_PREVIOUS_SHA256
    schema: str = RELEASE_CHAIN_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.release_id, str) or not self.release_id.strip():
            raise ReleaseChainError("release_id must be a non-empty string")
        object.__setattr__(
            self, "receipt_sha256", _require_sha256(self.receipt_sha256, "receipt_sha256")
        )
        object.__setattr__(
            self,
            "native_model_manifest_sha256",
            _require_sha256(
                self.native_model_manifest_sha256, "native_model_manifest_sha256"
            ),
        )
        object.__setattr__(
            self,
            "previous_entry_sha256",
            _require_sha256(self.previous_entry_sha256, "previous_entry_sha256"),
        )
        if self.schema != RELEASE_CHAIN_SCHEMA:
            raise ReleaseChainError("unsupported release-chain schema")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def entry_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


def append_release(
    entries: tuple[ReleaseChainEntry, ...] | list[ReleaseChainEntry],
    *,
    release_id: str,
    receipt_sha256: str,
    native_model_manifest_sha256: str,
) -> tuple[ReleaseChainEntry, ...]:
    """Append one release while preserving validated predecessor linkage."""

    current = tuple(entries)
    verify_release_chain(current)
    previous = current[-1].entry_sha256 if current else GENESIS_PREVIOUS_SHA256
    entry = ReleaseChainEntry(
        release_id=release_id,
        receipt_sha256=receipt_sha256,
        native_model_manifest_sha256=native_model_manifest_sha256,
        previous_entry_sha256=previous,
    )
    return (*current, entry)


def verify_release_chain(entries: tuple[ReleaseChainEntry, ...] | list[ReleaseChainEntry]) -> None:
    """Validate schema, uniqueness, and every predecessor hash in order."""

    previous = GENESIS_PREVIOUS_SHA256
    seen_release_ids: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, ReleaseChainEntry):
            raise TypeError("release chain entries must be ReleaseChainEntry instances")
        release_id = entry.release_id.strip()
        if release_id in seen_release_ids:
            raise ReleaseChainError(f"duplicate release_id in release chain: {release_id}")
        if entry.previous_entry_sha256 != previous:
            raise ReleaseChainError(
                f"release chain predecessor mismatch at index {index}: {release_id}"
            )
        seen_release_ids.add(release_id)
        previous = entry.entry_sha256


def save_release_chain(
    entries: tuple[ReleaseChainEntry, ...] | list[ReleaseChainEntry], path: str | Path
) -> Path:
    """Atomically persist a verified release chain with a document integrity hash."""

    current = tuple(entries)
    verify_release_chain(current)
    payload = [entry.as_dict() for entry in current]
    document = {
        "schema": RELEASE_CHAIN_SCHEMA,
        "entries": payload,
        "chain_sha256": _canonical_sha256(payload),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
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


def load_release_chain(path: str | Path) -> tuple[ReleaseChainEntry, ...]:
    """Load and verify a persisted release chain, failing closed on any tampering."""

    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseChainError(f"cannot read release chain: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != RELEASE_CHAIN_SCHEMA:
        raise ReleaseChainError("invalid or unsupported release-chain document")
    payload = document.get("entries")
    recorded_hash = document.get("chain_sha256")
    if not isinstance(payload, list):
        raise ReleaseChainError("release-chain entries must be a list")
    if not isinstance(recorded_hash, str) or _canonical_sha256(payload) != recorded_hash:
        raise ReleaseChainError("release-chain integrity hash mismatch")
    try:
        entries = tuple(ReleaseChainEntry(**item) for item in payload)
    except (TypeError, ValueError) as error:
        raise ReleaseChainError(f"invalid release-chain entry: {error}") from error
    verify_release_chain(entries)
    return entries


__all__ = [
    "GENESIS_PREVIOUS_SHA256",
    "RELEASE_CHAIN_SCHEMA",
    "ReleaseChainEntry",
    "ReleaseChainError",
    "append_release",
    "load_release_chain",
    "save_release_chain",
    "verify_release_chain",
]
