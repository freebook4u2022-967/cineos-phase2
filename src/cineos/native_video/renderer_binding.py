"""Bind complete-film rendering to the active native temporal state.

``FilmOrchestrator`` intentionally knows nothing about recurrent tensors.  The
continuity bridge owns shot-attempt state, while this adapter guarantees that a
native shot renderer receives that exact active state for the attempt being
rendered.  A native renderer therefore cannot silently bypass scene continuity
when used through this integration path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

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

    ``NativeFilmContinuityBridge.start_attempt`` must run before ``render``.  The
    complete-film orchestrator already guarantees that ordering when constructed
    with ``continuity.orchestrator_kwargs()``.  Failing closed here is deliberate:
    rendering without an active state would produce a visually valid file while
    silently dropping the long-range continuity contract.
    """

    renderer: NativeTemporalShotRenderer
    continuity: NativeFilmContinuityBridge

    def render(self, planned: Any, target: str | Path) -> Path:
        shot_id = str(getattr(planned, "shot_id", ""))
        if not shot_id:
            raise ValueError("planned shot requires a shot_id")
        state = self.continuity.state_for(shot_id)
        result = self.renderer.render(
            planned,
            target,
            temporal_state=state,
        )
        return Path(result)

    def cancel_pending(self) -> None:
        """Forward cooperative cancellation when the underlying renderer supports it."""
        cancel = getattr(self.renderer, "cancel_pending", None)
        if callable(cancel):
            cancel()
            return
        cancel = getattr(self.renderer, "cancel", None)
        if callable(cancel):
            cancel()
