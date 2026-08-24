"""Durable, versioned checkpoints for resumable film builds.

Checkpoints are intentionally backend-neutral.  The canonical ``FilmBuild`` state
remains the compatibility surface, while an optional integrity-protected runtime
state can carry renderer-owned resumable data such as native scene continuity
memory.  Writes are atomic on a single filesystem so a process interruption
cannot leave a partially-written checkpoint that later masquerades as valid
state.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .build import BuildStatus, FilmBuild
from .shot_state import ShotState

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(ValueError):
    """Raised when a checkpoint is malformed or uses an unsupported schema."""


def _canonical_hash(payload: Any) -> str:
    """Return a deterministic SHA-256 for JSON-safe checkpoint state."""
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("checkpoint runtime_state must be JSON-serializable") from error
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_payload(
    build: FilmBuild,
    *,
    runtime_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the stable, versioned payload persisted for ``build``.

    ``runtime_state`` is deliberately optional so existing callers and old
    checkpoints remain valid.  When present it is hashed independently from the
    build content hash, preventing continuity/checkpoint corruption from being
    silently accepted during resume.
    """
    if not isinstance(build, FilmBuild):
        raise TypeError("build must be a FilmBuild")
    if runtime_state is not None and not isinstance(runtime_state, dict):
        raise TypeError("runtime_state must be a mapping")

    payload = asdict(build)
    payload["status"] = str(build.status)
    document: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "build": payload,
        "content_hash": build.content_hash,
    }
    if runtime_state is not None:
        document["runtime_state"] = runtime_state
        document["runtime_state_hash"] = _canonical_hash(runtime_state)
    return document


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync after atomic replacement.

    ``os.replace`` protects readers from partial checkpoint contents, but on POSIX
    filesystems durability across a sudden power loss also requires syncing the
    containing directory entry. Some platforms do not permit opening directories;
    those platforms retain the atomic-replace guarantee and safely skip this step.
    """
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        # Directory fsync is not supported on every filesystem/platform.
        pass
    finally:
        os.close(fd)


def save_checkpoint(
    build: FilmBuild,
    path: str | Path,
    *,
    runtime_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically persist a build checkpoint and return its final path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            checkpoint_payload(build, runtime_state=runtime_state),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

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
        _fsync_directory(target.parent)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return target


def _load_document(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointError(f"cannot read checkpoint {source}: {error}") from error

    if not isinstance(document, dict):
        raise CheckpointError("checkpoint root must be an object")
    version = document.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint schema {version!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    return source, document


def load_checkpoint_runtime_state(path: str | Path) -> dict[str, Any] | None:
    """Load optional runtime state after verifying its independent integrity hash.

    Old checkpoints legitimately return ``None``.  A checkpoint that contains
    only one half of the runtime-state/hash pair is malformed and rejected.
    """
    _, document = _load_document(path)
    raw = document.get("runtime_state")
    recorded_hash = document.get("runtime_state_hash")
    if raw is None and recorded_hash is None:
        return None
    if not isinstance(raw, dict):
        raise CheckpointError("checkpoint runtime_state must be an object")
    if not isinstance(recorded_hash, str) or not recorded_hash:
        raise CheckpointError("checkpoint runtime_state is missing integrity hash")
    if _canonical_hash(raw) != recorded_hash:
        raise CheckpointError("checkpoint runtime_state hash mismatch")
    return raw


def load_checkpoint(path: str | Path) -> FilmBuild:
    """Load and validate a ``FilmBuild`` from a versioned checkpoint."""
    _, document = _load_document(path)
    raw = document.get("build")
    if not isinstance(raw, dict):
        raise CheckpointError("checkpoint is missing build state")

    recorded_hash = document.get("content_hash")
    if not isinstance(recorded_hash, str) or not recorded_hash:
        raise CheckpointError("checkpoint is missing content hash")

    data = dict(raw)
    try:
        data["status"] = BuildStatus(data["status"])
        shot_states = data.get("shot_states", [])
        if not isinstance(shot_states, list):
            raise TypeError("shot_states must be a list")
        if any(not isinstance(item, dict) for item in shot_states):
            raise TypeError("shot_states entries must be objects")
        data["shot_states"] = [ShotState(**item) for item in shot_states]
        build = FilmBuild(**data)
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError(f"invalid checkpoint build state: {error}") from error

    if recorded_hash != build.content_hash:
        raise CheckpointError("checkpoint content hash mismatch")
    return build
