"""Bind complete-film rendering to the active native temporal state.

``FilmOrchestrator`` intentionally knows nothing about recurrent tensors. The
continuity bridge owns shot-attempt state, while this adapter guarantees that a
native shot renderer receives that exact active state for the attempt being
rendered. A native renderer therefore cannot silently bypass scene continuity
when used through this integration path.

The binding also records artifact-integrity metadata on the active temporal
state. This gives accepted scene anchors a durable cryptographic link to the
actual native artifact that produced them, which is useful for checkpoint resume,
audit, and future cache/reuse decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .artifact_integrity import provenance_for
from .film_bridge import NativeFilmContinuityBridge
from .temporal_model import TemporalSequenceState


class NativeTemporalShotRenderer(Protocol):
    """Native shot renderer contract with explicit temporal-state ownership."""

    def render(
        self,
        planned: Any,
        target: str | Path,
        *,
        temporal_state: TemporalSequenceState,
    ) -> str | Path: ...


@dataclass(slots=True)
class NativeFilmRendererBinding:
    """Inject the active continuity state into a native shot renderer.

    ``NativeFilmContinuityBridge.start_attempt`` must run before ``render``. The
    complete-film orchestrator already guarantees that ordering when constructed
    with ``continuity.orchestrator_kwargs()``. Failing closed here is deliberate:
    rendering without an active state would produce a visually valid file while
    silently dropping the long-range continuity contract.

    A renderer result is accepted by this boundary only when it names a real,
    non-empty file. Its byte size and SHA-256 digest are then attached to the active
    attempt state. Rejected attempts are discarded by the continuity bridge; an
    accepted attempt therefore promotes both visual state and artifact provenance
    into durable scene memory atomically at the orchestration level.
    """

    renderer: NativeTemporalShotRenderer
    continuity: NativeFilmContinuityBridge

    def render(self, planned: Any, target: str | Path) -> Path:
        shot_id = str(getattr(planned, "shot_id", ""))
        if not shot_id:
            raise ValueError("planned shot requires a shot_id")
        state = self.continuity.state_for(shot_id)
        result = Path(
            self.renderer.render(
                planned,
                target,
                temporal_state=state,
            )
        )
        provenance = provenance_for(result)
        state.metadata["native_artifact_sha256"] = provenance.sha256
        state.metadata["native_artifact_bytes"] = provenance.byte_size
        return result

    def cancel_pending(self) -> None:
        """Forward cooperative cancellation when the underlying renderer supports it."""
        cancel = getattr(self.renderer, "cancel_pending", None)
        if callable(cancel):
            cancel()
            return
        cancel = getattr(self.renderer, "cancel", None)
        if callable(cancel):
            cancel()
