"""Durable workspace for resumable native temporal shot renders.

A complete-film checkpoint can resume between shots, but long native shots also
need a crash-safe boundary *inside* a shot.  This module owns that boundary.  It
persists a versioned render manifest, integrity-checked temporal state, and the
already-decoded frame sequence in one deterministic workspace.  A later process
may resume only when the immutable render contract still matches; stale or
partially corrupted workspaces fail closed instead of silently poisoning scene
continuity.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .temporal_checkpoint import (
    TemporalCheckpointError,
    restore_temporal_checkpoint,
    temporal_checkpoint_payload,
)
from .temporal_model import TemporalSequenceState

RENDER_WORKSPACE_SCHEMA_VERSION = 1


class RenderWorkspaceError(ValueError):
    """Raised when a resumable native render workspace is invalid or incompatible."""


def _canonical_json_hash(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RenderWorkspaceError(
            "render workspace contract must be finite JSON data"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(path: Path) -> None:
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
        pass
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class RenderContract:
    """Immutable facts that must remain identical across a resumed shot render."""

    shot_id: str
    base_state_hash: str
    width: int
    height: int
    fps: int
    frame_count: int
    decoder_id: str
    renderer_id: str
    conditioning_hash: str

    def __post_init__(self) -> None:
        if not self.shot_id:
            raise ValueError("render contract requires a shot_id")
        if not self.base_state_hash:
            raise ValueError("render contract requires a base_state_hash")
        if min(self.width, self.height, self.fps, self.frame_count) <= 0:
            raise ValueError(
                "render contract dimensions, fps and frame_count must be positive"
            )
        if not self.decoder_id or not self.renderer_id:
            raise ValueError(
                "render contract requires decoder and renderer identifiers"
            )
        if not self.conditioning_hash:
            raise ValueError("render contract requires a conditioning_hash")

    def payload(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "base_state_hash": self.base_state_hash,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "decoder_id": self.decoder_id,
            "renderer_id": self.renderer_id,
            "conditioning_hash": self.conditioning_hash,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_json_hash(self.payload())


@dataclass(slots=True)
class NativeRenderWorkspace:
    """Crash-safe persistent state for one native temporal shot attempt."""

    root: Path
    contract: RenderContract

    @classmethod
    def open(cls, root: str | Path, contract: RenderContract) -> NativeRenderWorkspace:
        workspace = cls(Path(root), contract)
        workspace.root.mkdir(parents=True, exist_ok=True)
        workspace.frames_dir.mkdir(parents=True, exist_ok=True)
        if workspace.manifest_path.exists():
            workspace._validate_manifest()
        else:
            workspace._write_manifest()
        return workspace

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "temporal-checkpoint.json"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    def frame_path(self, frame_index: int) -> Path:
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        return self.frames_dir / f"frame-{frame_index:06d}.ppm"

    def _manifest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": RENDER_WORKSPACE_SCHEMA_VERSION,
            "contract": self.contract.payload(),
            "contract_hash": self.contract.fingerprint,
        }

    def _write_manifest(self) -> None:
        _atomic_write_json(self.manifest_path, self._manifest_payload())

    def _validate_manifest(self) -> None:
        try:
            document = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RenderWorkspaceError(
                f"cannot read render workspace manifest: {error}"
            ) from error
        if not isinstance(document, dict):
            raise RenderWorkspaceError(
                "render workspace manifest root must be an object"
            )
        if document.get("schema_version") != RENDER_WORKSPACE_SCHEMA_VERSION:
            raise RenderWorkspaceError("unsupported render workspace schema")
        raw_contract = document.get("contract")
        recorded_hash = document.get("contract_hash")
        if not isinstance(raw_contract, dict) or not isinstance(recorded_hash, str):
            raise RenderWorkspaceError("render workspace manifest is incomplete")
        if _canonical_json_hash(raw_contract) != recorded_hash:
            raise RenderWorkspaceError("render workspace contract hash mismatch")
        if (
            recorded_hash != self.contract.fingerprint
            or raw_contract != self.contract.payload()
        ):
            raise RenderWorkspaceError(
                "render workspace belongs to a different render contract"
            )

    def save_state(self, state: TemporalSequenceState) -> Path:
        if state.shot_id != self.contract.shot_id:
            raise RenderWorkspaceError("temporal state belongs to a different shot")
        _atomic_write_json(self.checkpoint_path, temporal_checkpoint_payload(state))
        return self.checkpoint_path

    def load_state(self) -> TemporalSequenceState | None:
        if not self.checkpoint_path.exists():
            return None
        try:
            document = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RenderWorkspaceError(
                f"cannot read temporal render checkpoint: {error}"
            ) from error
        try:
            state = restore_temporal_checkpoint(document)
        except TemporalCheckpointError as error:
            raise RenderWorkspaceError(
                f"invalid temporal render checkpoint: {error}"
            ) from error
        if state.shot_id != self.contract.shot_id:
            raise RenderWorkspaceError(
                "temporal render checkpoint belongs to a different shot"
            )
        return state

    def next_frame_index(self) -> int:
        """Return first frame allowed after durable-frame validation."""
        state = self.load_state()
        if state is None:
            existing = tuple(self.frames_dir.glob("frame-*.ppm"))
            if existing:
                raise RenderWorkspaceError(
                    "render workspace has frames but no temporal checkpoint"
                )
            return 0

        next_index = state.last_frame_index + 1
        if next_index < 0 or next_index > self.contract.frame_count:
            raise RenderWorkspaceError(
                "temporal checkpoint frame index exceeds render contract"
            )

        for index in range(next_index):
            frame = self.frame_path(index)
            if not frame.is_file() or frame.stat().st_size <= 0:
                raise RenderWorkspaceError(
                    f"render workspace is missing committed frame {index}"
                )
        unexpected = self.frame_path(next_index)
        if next_index < self.contract.frame_count and unexpected.exists():
            raise RenderWorkspaceError(
                "render workspace contains an uncommitted future frame"
            )
        return next_index

    def commit_frame(
        self, frame_index: int, rgb_ppm: bytes, state: TemporalSequenceState
    ) -> Path:
        """Atomically persist one decoded frame, then checkpoint the matching state.

        The frame is promoted first and the checkpoint second.  A crash between the
        two writes leaves an uncommitted future frame, which ``next_frame_index``
        detects and rejects rather than treating as accepted continuity.
        """
        if frame_index < 0 or frame_index >= self.contract.frame_count:
            raise RenderWorkspaceError("frame index is outside render contract")
        if state.last_frame_index != frame_index:
            raise RenderWorkspaceError(
                "temporal state does not commit the requested frame"
            )
        target = self.frame_path(frame_index)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(rgb_ppm)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
            _fsync_directory(target.parent)
        except Exception:
            temp.unlink(missing_ok=True)
            raise
        self.save_state(state)
        return target

    def clear(self) -> None:
        """Remove durable attempt artifacts after successful external promotion."""
        for frame in self.frames_dir.glob("frame-*.ppm"):
            frame.unlink(missing_ok=True)
        self.checkpoint_path.unlink(missing_ok=True)
        self.manifest_path.unlink(missing_ok=True)
        try:
            self.frames_dir.rmdir()
        except OSError:
            pass
        try:
            self.root.rmdir()
        except OSError:
            pass
