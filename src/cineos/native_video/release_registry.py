"""Crash-safe authenticated production release registry for CINEOS.

Release-chain hashes and detached HMAC seals establish provenance, but production
also needs an atomic activation boundary. Writing ``release-chain.json`` and its
seal in place can expose a torn pair if a process or host fails between writes.

This registry stores immutable, authenticated snapshots and activates one snapshot
with a single atomic ``CURRENT`` pointer replacement. A failed update therefore
leaves the previously active release fully readable and authenticated. Snapshot
identifiers are content-derived and bind both the chain bytes and signing key id,
which also makes explicit key rotation safe.

Authentication alone cannot distinguish a legitimate historical snapshot from an
attacker-induced rollback of ``CURRENT``. Callers that persist the last trusted
active generation can therefore pass ``expected_generation_id`` when loading. The
registry then fails closed before accepting any different (even otherwise valid)
snapshot.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .release_chain import ReleaseChainEntry, load_release_chain, save_release_chain
from .release_seal import (
    ReleaseChainSeal,
    seal_release_chain_file,
    verify_release_chain_file,
)

RELEASE_REGISTRY_SCHEMA = "cineos-release-registry/0.1"
ACTIVE_SNAPSHOT_FILE = "CURRENT"
SNAPSHOTS_DIRECTORY = "snapshots"
CHAIN_FILE = "release-chain.json"
SEAL_FILE = "release-chain.seal.json"


class ReleaseRegistryError(ValueError):
    """Raised when the active production release registry is invalid."""


@dataclass(frozen=True, slots=True)
class VerifiedReleaseSnapshot:
    """One active registry snapshot after chain and seal authentication."""

    generation_id: str
    path: Path
    entries: tuple[ReleaseChainEntry, ...]
    seal: ReleaseChainSeal
    schema: str = RELEASE_REGISTRY_SCHEMA


def _generation_id(chain_bytes: bytes, key_id: str) -> str:
    payload = chain_bytes + b"\x00" + key_id.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_generation_id(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ReleaseRegistryError("generation id must be one SHA-256 hex digest")
    return normalized


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability on platforms that expose directory fsync."""

    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def commit_release_snapshot(
    entries: tuple[ReleaseChainEntry, ...] | list[ReleaseChainEntry],
    root: str | Path,
    *,
    key: bytes | bytearray | memoryview,
    key_id: str,
) -> VerifiedReleaseSnapshot:
    """Persist and atomically activate an authenticated immutable release snapshot.

    The old ``CURRENT`` pointer is not changed until the new chain and seal have
    both been persisted. If any earlier step fails, readers continue to observe
    the previous active snapshot.
    """

    registry_root = Path(root)
    snapshots_root = registry_root / SNAPSHOTS_DIRECTORY
    snapshots_root.mkdir(parents=True, exist_ok=True)

    stage: Path | None = Path(
        tempfile.mkdtemp(prefix=".staging-", dir=str(snapshots_root))
    )
    try:
        chain_path = save_release_chain(entries, stage / CHAIN_FILE)
        seal = seal_release_chain_file(
            chain_path,
            stage / SEAL_FILE,
            key=key,
            key_id=key_id,
        )
        chain_bytes = chain_path.read_bytes()
        generation_id = _generation_id(chain_bytes, seal.key_id)
        destination = snapshots_root / generation_id

        if destination.exists():
            # Content-derived generations are reusable, but only after full
            # authentication with the supplied key. Never silently trust an
            # existing directory with the same name.
            shutil.rmtree(stage)
            stage = None
            verify_release_chain_file(
                destination / CHAIN_FILE,
                destination / SEAL_FILE,
                key=key,
                expected_key_id=seal.key_id,
            )
        else:
            os.replace(stage, destination)
            stage = None
            _fsync_directory(snapshots_root)

        _atomic_write_text(registry_root / ACTIVE_SNAPSHOT_FILE, generation_id + "\n")
        return load_verified_release_snapshot(
            registry_root,
            key=key,
            expected_key_id=seal.key_id,
            expected_generation_id=generation_id,
        )
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def load_verified_release_snapshot(
    root: str | Path,
    *,
    key: bytes | bytearray | memoryview,
    expected_key_id: str | None = None,
    expected_generation_id: str | None = None,
) -> VerifiedReleaseSnapshot:
    """Load the active snapshot and fail closed on tampering or trusted rollback.

    ``expected_generation_id`` is an optional caller-held trust anchor. It should
    come from state protected independently of this registry (for example a
    deployment database, KMS metadata, or release controller). Supplying it makes
    a rollback of ``CURRENT`` to an older but correctly signed snapshot detectable.
    """

    registry_root = Path(root)
    pointer = registry_root / ACTIVE_SNAPSHOT_FILE
    try:
        generation_id = _validate_generation_id(pointer.read_text(encoding="utf-8"))
    except OSError as error:
        raise ReleaseRegistryError(
            f"cannot read active release pointer: {error}"
        ) from error

    if expected_generation_id is not None:
        trusted_generation_id = _validate_generation_id(expected_generation_id)
        if generation_id != trusted_generation_id:
            raise ReleaseRegistryError(
                "active release generation differs from trusted generation"
            )

    snapshot_path = registry_root / SNAPSHOTS_DIRECTORY / generation_id
    chain_path = snapshot_path / CHAIN_FILE
    seal_path = snapshot_path / SEAL_FILE
    try:
        seal = verify_release_chain_file(
            chain_path,
            seal_path,
            key=key,
            expected_key_id=expected_key_id,
        )
        chain_bytes = chain_path.read_bytes()
        entries = load_release_chain(chain_path)
    except (OSError, ValueError) as error:
        raise ReleaseRegistryError(
            f"active release snapshot failed authentication: {error}"
        ) from error

    actual_generation_id = _generation_id(chain_bytes, seal.key_id)
    if actual_generation_id != generation_id:
        raise ReleaseRegistryError("active release generation id mismatch")

    return VerifiedReleaseSnapshot(
        generation_id=generation_id,
        path=snapshot_path,
        entries=entries,
        seal=seal,
    )


__all__ = [
    "ACTIVE_SNAPSHOT_FILE",
    "CHAIN_FILE",
    "RELEASE_REGISTRY_SCHEMA",
    "SEAL_FILE",
    "SNAPSHOTS_DIRECTORY",
    "ReleaseRegistryError",
    "VerifiedReleaseSnapshot",
    "commit_release_snapshot",
    "load_verified_release_snapshot",
]
