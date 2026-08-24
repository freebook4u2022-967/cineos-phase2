"""Durable, versioned checkpoints for resumable film builds.

Checkpoints are intentionally backend-neutral and contain only serializable build
state. Writes are atomic on a single filesystem so a process interruption cannot
leave a partially-written checkpoint that later masquerades as valid state.
"""

from __future__ import annotations

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


def checkpoint_payload(build: FilmBuild) -> dict[str, Any]:
    """Return the stable, versioned payload persisted for ``build``."""
    payload = asdict(build)
    payload["status"] = str(build.status)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "build": payload,
        "content_hash": build.content_hash,
    }


def save_checkpoint(build: FilmBuild, path: str | Path) -> Path:
    """Atomically persist a build checkpoint and return its final path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        checkpoint_payload(build),
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    ) + "\n"

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


def load_checkpoint(path: str | Path) -> FilmBuild:
    """Load and validate a ``FilmBuild`` from a versioned checkpoint."""
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
    raw = document.get("build")
    if not isinstance(raw, dict):
        raise CheckpointError("checkpoint is missing build state")

    data = dict(raw)
    try:
        data["status"] = BuildStatus(data["status"])
        shot_states = data.get("shot_states", [])
        if not isinstance(shot_states, list):
            raise TypeError("shot_states must be a list")
        data["shot_states"] = [ShotState(**item) for item in shot_states]
        build = FilmBuild(**data)
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError(f"invalid checkpoint build state: {error}") from error

    recorded_hash = document.get("content_hash")
    if recorded_hash and recorded_hash != build.content_hash:
        raise CheckpointError("checkpoint content hash mismatch")
    return build
