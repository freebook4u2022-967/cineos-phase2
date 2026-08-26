"""Versioned, integrity-checked checkpoints for native temporal sequences.

Film checkpoints already provide durable build-level recovery.  This module gives
native video sequence state its own compatibility and corruption boundary so a
long render can safely persist, resume, migrate, or reject temporal memory
without silently poisoning continuity.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .temporal_model import TemporalSequenceState

TEMPORAL_CHECKPOINT_SCHEMA_VERSION = 1


class TemporalCheckpointError(ValueError):
    """Raised when native temporal checkpoint state is malformed or unsupported."""


def _canonical_hash(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TemporalCheckpointError(
            "temporal checkpoint state must be finite JSON data"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def temporal_checkpoint_payload(state: TemporalSequenceState) -> dict[str, Any]:
    """Return a stable, versioned checkpoint document for ``state``."""
    if not isinstance(state, TemporalSequenceState):
        raise TypeError("state must be a TemporalSequenceState")
    if not state.shot_id:
        raise TemporalCheckpointError("temporal checkpoint requires a shot_id")

    snapshot = state.snapshot()
    return {
        "schema_version": TEMPORAL_CHECKPOINT_SCHEMA_VERSION,
        "state": snapshot,
        "state_hash": _canonical_hash(snapshot),
    }


def restore_temporal_checkpoint(document: dict[str, Any]) -> TemporalSequenceState:
    """Restore and validate a native temporal sequence checkpoint.

    The schema is intentionally strict.  Unsupported future versions are rejected
    rather than guessed, and state is hashed before construction so corrupted
    continuity memory cannot be resumed as if it were valid.
    """
    if not isinstance(document, dict):
        raise TemporalCheckpointError("temporal checkpoint root must be an object")

    version = document.get("schema_version")
    if version != TEMPORAL_CHECKPOINT_SCHEMA_VERSION:
        raise TemporalCheckpointError(
            f"unsupported temporal checkpoint schema {version!r}; "
            f"expected {TEMPORAL_CHECKPOINT_SCHEMA_VERSION}"
        )

    raw_state = document.get("state")
    if not isinstance(raw_state, dict):
        raise TemporalCheckpointError("temporal checkpoint is missing state")

    recorded_hash = document.get("state_hash")
    if not isinstance(recorded_hash, str) or not recorded_hash:
        raise TemporalCheckpointError("temporal checkpoint is missing state hash")
    if _canonical_hash(raw_state) != recorded_hash:
        raise TemporalCheckpointError("temporal checkpoint state hash mismatch")

    try:
        state = TemporalSequenceState.restore(raw_state)
    except (TypeError, ValueError) as error:
        raise TemporalCheckpointError(
            f"invalid temporal checkpoint state: {error}"
        ) from error

    if not state.shot_id:
        raise TemporalCheckpointError("temporal checkpoint state requires a shot_id")
    if state.last_frame_index < -1:
        raise TemporalCheckpointError("temporal checkpoint frame index is invalid")
    if state.last_frame_index == -1 and state.last_latent is not None:
        raise TemporalCheckpointError(
            "unstarted temporal checkpoint cannot contain a last latent"
        )
    if state.last_frame_index >= 0 and state.last_latent is None:
        raise TemporalCheckpointError(
            "started temporal checkpoint must contain a last latent"
        )

    frames_generated = state.metadata.get("frames_generated")
    if frames_generated is not None:
        if not isinstance(frames_generated, int) or isinstance(frames_generated, bool):
            raise TemporalCheckpointError("frames_generated metadata must be an integer")
        expected = state.last_frame_index + 1
        if frames_generated != expected:
            raise TemporalCheckpointError(
                "frames_generated metadata does not match temporal frame index"
            )

    accepted = state.metadata.get("accepted_candidates")
    if accepted is not None:
        if not isinstance(accepted, int) or isinstance(accepted, bool) or accepted < 0:
            raise TemporalCheckpointError(
                "accepted_candidates metadata must be a non-negative integer"
            )
        minimum = max(state.last_frame_index + 1, 0)
        if accepted < minimum:
            raise TemporalCheckpointError(
                "accepted_candidates metadata cannot trail committed frames"
            )

    return state
