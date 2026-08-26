"""Authenticated recovery for interrupted CINEOS release activation.

Release activation intentionally leaves ``.ACTIVATION.lock`` in place when a
process or host disappears mid-activation.  Automatic lock breaking would trade
availability for the possibility of two production writers, so recovery is an
explicit operator action with two trust checks:

* the caller must present the exact lock contents it previously observed; and
* the currently active release must authenticate against a caller-held trusted
  generation before the lock is removed.

This keeps normal activation fail-closed while providing a deterministic,
auditable path to restore availability after an interrupted deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .release_registry import (
    ACTIVATION_LOCK_FILE,
    ReleaseRegistryError,
    VerifiedReleaseSnapshot,
    load_verified_release_snapshot,
)


@dataclass(frozen=True, slots=True)
class ReleaseLockRecovery:
    """Evidence returned after a stale activation lock is safely removed."""

    lock_contents: str
    verified_snapshot: VerifiedReleaseSnapshot


def read_activation_lock(root: str | Path) -> str:
    """Read the current activation-lock payload without changing registry state."""

    path = Path(root) / ACTIVATION_LOCK_FILE
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ReleaseRegistryError("release activation lock is missing") from error
    except OSError as error:
        raise ReleaseRegistryError(
            f"cannot read release activation lock: {error}"
        ) from error
    if not contents:
        raise ReleaseRegistryError("release activation lock is empty")
    return contents


def recover_activation_lock(
    root: str | Path,
    *,
    key: bytes | bytearray | memoryview,
    expected_generation_id: str,
    expected_lock_contents: str,
    expected_key_id: str | None = None,
) -> ReleaseLockRecovery:
    """Remove one observed stale activation lock after authenticating production.

    ``expected_lock_contents`` is a compare-and-swap token supplied by the
    recovery controller.  If the lock no longer contains exactly those bytes,
    recovery fails closed rather than deleting a lock created or changed after
    inspection.  ``expected_generation_id`` must be held independently of the
    registry and is verified through :func:`load_verified_release_snapshot`
    before any deletion occurs.
    """

    registry_root = Path(root)
    observed = read_activation_lock(registry_root)
    if observed != expected_lock_contents:
        raise ReleaseRegistryError(
            "release activation lock changed since recovery inspection"
        )

    verified = load_verified_release_snapshot(
        registry_root,
        key=key,
        expected_key_id=expected_key_id,
        expected_generation_id=expected_generation_id,
    )

    # Re-read immediately before unlinking.  A cooperative writer cannot replace
    # an existing lock, but this second check protects against external operator
    # or filesystem intervention between authentication and recovery.
    current = read_activation_lock(registry_root)
    if current != expected_lock_contents:
        raise ReleaseRegistryError("release activation lock changed during recovery")

    lock_path = registry_root / ACTIVATION_LOCK_FILE
    try:
        lock_path.unlink()
    except FileNotFoundError as error:
        raise ReleaseRegistryError(
            "release activation lock disappeared during recovery"
        ) from error
    except OSError as error:
        raise ReleaseRegistryError(
            f"cannot remove release activation lock: {error}"
        ) from error

    # Persist the directory entry removal where supported.  Recovery should not
    # report success while the stale lock deletion is only in page cache.
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(registry_root, flags)
    except OSError:
        descriptor = None
    if descriptor is not None:
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    return ReleaseLockRecovery(
        lock_contents=expected_lock_contents,
        verified_snapshot=verified,
    )


__all__ = [
    "ReleaseLockRecovery",
    "read_activation_lock",
    "recover_activation_lock",
]
