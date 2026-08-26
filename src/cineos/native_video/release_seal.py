"""Authenticated integrity seals for persisted CINEOS production release chains.

A hash chain alone detects accidental corruption, but an attacker who can rewrite the
chain file can also recompute unkeyed hashes. This module adds a detached HMAC-SHA256
seal whose secret key is supplied by the deployment environment (or a future KMS)
and is never persisted with the release metadata.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RELEASE_SEAL_SCHEMA = "cineos-release-seal/0.1"
MIN_HMAC_KEY_BYTES = 32


class ReleaseSealError(ValueError):
    """Raised when a release seal is malformed, mismatched, or unauthenticated."""


def _require_key(key: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(key, (bytes, bytearray, memoryview)):
        raise TypeError("release seal key must be bytes-like")
    normalized = bytes(key)
    if len(normalized) < MIN_HMAC_KEY_BYTES:
        raise ReleaseSealError(
            f"release seal key must contain at least {MIN_HMAC_KEY_BYTES} bytes"
        )
    return normalized


def _require_key_id(key_id: str) -> str:
    if not isinstance(key_id, str) or not key_id.strip():
        raise ReleaseSealError("key_id must be a non-empty string")
    return key_id.strip()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hmac_sha256(key: bytes, payload: bytes) -> str:
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


@dataclass(frozen=True, slots=True)
class ReleaseChainSeal:
    """Detached authenticated seal for one exact persisted release-chain document."""

    key_id: str
    chain_sha256: str
    hmac_sha256: str
    schema: str = RELEASE_SEAL_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_id", _require_key_id(self.key_id))
        for name in ("chain_sha256", "hmac_sha256"):
            value = str(getattr(self, name)).strip().lower()
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ReleaseSealError(f"{name} must be a 64-character SHA-256 hex digest")
            object.__setattr__(self, name, value)
        if self.schema != RELEASE_SEAL_SCHEMA:
            raise ReleaseSealError("unsupported release-seal schema")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "key_id": self.key_id,
            "chain_sha256": self.chain_sha256,
            "hmac_sha256": self.hmac_sha256,
        }


def build_release_chain_seal(
    chain_bytes: bytes,
    *,
    key: bytes | bytearray | memoryview,
    key_id: str,
) -> ReleaseChainSeal:
    """Authenticate the exact bytes of a persisted release-chain document."""

    if not isinstance(chain_bytes, bytes):
        raise TypeError("chain_bytes must be bytes")
    secret = _require_key(key)
    return ReleaseChainSeal(
        key_id=_require_key_id(key_id),
        chain_sha256=_sha256_bytes(chain_bytes),
        hmac_sha256=_hmac_sha256(secret, chain_bytes),
    )


def verify_release_chain_seal(
    chain_bytes: bytes,
    seal: ReleaseChainSeal,
    *,
    key: bytes | bytearray | memoryview,
    expected_key_id: str | None = None,
) -> None:
    """Fail closed unless the chain bytes match both digest and keyed authentication."""

    if not isinstance(chain_bytes, bytes):
        raise TypeError("chain_bytes must be bytes")
    if not isinstance(seal, ReleaseChainSeal):
        raise TypeError("seal must be a ReleaseChainSeal")
    secret = _require_key(key)
    if expected_key_id is not None and seal.key_id != _require_key_id(expected_key_id):
        raise ReleaseSealError("release seal key_id mismatch")
    actual_chain_sha256 = _sha256_bytes(chain_bytes)
    if not hmac.compare_digest(actual_chain_sha256, seal.chain_sha256):
        raise ReleaseSealError("release-chain digest mismatch")
    actual_hmac = _hmac_sha256(secret, chain_bytes)
    if not hmac.compare_digest(actual_hmac, seal.hmac_sha256):
        raise ReleaseSealError("release-chain authentication failed")


def save_release_chain_seal(seal: ReleaseChainSeal, path: str | Path) -> Path:
    """Atomically persist a detached seal without ever persisting the signing key."""

    if not isinstance(seal, ReleaseChainSeal):
        raise TypeError("seal must be a ReleaseChainSeal")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(seal.as_dict(), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
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


def load_release_chain_seal(path: str | Path) -> ReleaseChainSeal:
    """Load a detached seal and validate its schema and digest fields."""

    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseSealError(f"cannot read release seal: {error}") from error
    if not isinstance(document, dict):
        raise ReleaseSealError("release seal must be a JSON object")
    try:
        return ReleaseChainSeal(
            key_id=document["key_id"],
            chain_sha256=document["chain_sha256"],
            hmac_sha256=document["hmac_sha256"],
            schema=document.get("schema", ""),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReleaseSealError(f"invalid release seal: {error}") from error


def seal_release_chain_file(
    chain_path: str | Path,
    seal_path: str | Path,
    *,
    key: bytes | bytearray | memoryview,
    key_id: str,
) -> ReleaseChainSeal:
    """Build and atomically persist a seal for an existing release-chain file."""

    source = Path(chain_path)
    try:
        chain_bytes = source.read_bytes()
    except OSError as error:
        raise ReleaseSealError(f"cannot read release chain for sealing: {error}") from error
    seal = build_release_chain_seal(chain_bytes, key=key, key_id=key_id)
    save_release_chain_seal(seal, seal_path)
    return seal


def verify_release_chain_file(
    chain_path: str | Path,
    seal_path: str | Path,
    *,
    key: bytes | bytearray | memoryview,
    expected_key_id: str | None = None,
) -> ReleaseChainSeal:
    """Authenticate a persisted release-chain file against its detached seal."""

    try:
        chain_bytes = Path(chain_path).read_bytes()
    except OSError as error:
        raise ReleaseSealError(f"cannot read release chain for verification: {error}") from error
    seal = load_release_chain_seal(seal_path)
    verify_release_chain_seal(
        chain_bytes,
        seal,
        key=key,
        expected_key_id=expected_key_id,
    )
    return seal


__all__ = [
    "MIN_HMAC_KEY_BYTES",
    "RELEASE_SEAL_SCHEMA",
    "ReleaseChainSeal",
    "ReleaseSealError",
    "build_release_chain_seal",
    "load_release_chain_seal",
    "save_release_chain_seal",
    "seal_release_chain_file",
    "verify_release_chain_file",
    "verify_release_chain_seal",
]
