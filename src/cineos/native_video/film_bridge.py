"""Bridge native temporal continuity into complete-film orchestration.

The bridge deliberately commits scene continuity only after whole-shot validation
passes. Each retry starts from the last durable accepted scene anchor, so a render
attempt that later fails film-level QC cannot poison the next shot or a resumed
production checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .scene_memory import SceneContinuityMemory, SceneTransitionPolicy
from .temporal_model import NativeTemporalModel, TemporalSequenceState

FILM_CONTINUITY_RUNTIME_KIND = "cineos-native-film-continuity/0.1"


@dataclass(slots=True)
class NativeFilmContinuityBridge:
    """Transactional adapter between FilmOrchestrator and native temporal state."""

    model: NativeTemporalModel
    memory: SceneContinuityMemory = field(default_factory=SceneContinuityMemory)
    device: str = "cpu"
    _active: dict[str, TemporalSequenceState] = field(default_factory=dict, init=False)

    @classmethod
    def default(cls, *, device: str = "cpu") -> NativeFilmContinuityBridge:
        return cls(model=NativeTemporalModel.initialized(), device=device)

    def _policy_for(self, planned: Any) -> SceneTransitionPolicy:
        payload = dict(getattr(planned, "payload", {}) or {})
        if bool(payload.get("hard_cut", False)) or bool(
            payload.get("continuity_reset", False)
        ):
            return SceneTransitionPolicy.hard_cut()
        hidden_carry = payload.get("continuity_hidden_carry")
        preserve_reference = payload.get("preserve_latent_reference")
        if hidden_carry is None and preserve_reference is None:
            return SceneTransitionPolicy()
        return SceneTransitionPolicy(
            hidden_carry=(
                SceneTransitionPolicy().hidden_carry
                if hidden_carry is None
                else float(hidden_carry)
            ),
            preserve_latent_reference=(
                SceneTransitionPolicy().preserve_latent_reference
                if preserve_reference is None
                else bool(preserve_reference)
            ),
        )

    def start_attempt(self, planned: Any, scene_index: int, attempt: int) -> None:
        """Create a fresh attempt state from only the last accepted scene anchor."""
        if attempt <= 0:
            raise ValueError("attempt must be positive")
        shot_id = str(getattr(planned, "shot_id", ""))
        if not shot_id:
            raise ValueError("planned shot requires a shot_id")
        self._active[shot_id] = self.memory.start_shot(
            self.model,
            scene_index=scene_index,
            shot_id=shot_id,
            device=self.device,
            policy=self._policy_for(planned),
        )
        self._active[shot_id].metadata["film_attempt"] = attempt

    def state_for(self, shot_id: str) -> TemporalSequenceState:
        """Return the mutable temporal state owned by the current render attempt."""
        try:
            return self._active[shot_id]
        except KeyError as error:
            raise KeyError(f"no active temporal state for shot {shot_id!r}") from error

    def accept_attempt(self, planned: Any, scene_index: int, attempt: int) -> None:
        """Promote the accepted attempt end-state into durable scene memory."""
        shot_id = str(getattr(planned, "shot_id", ""))
        state = self.state_for(shot_id)
        if int(state.metadata.get("film_attempt", -1)) != attempt:
            raise ValueError("accepted attempt does not match active temporal state")
        self.memory.record_accepted_shot(scene_index=scene_index, state=state)
        self._active.pop(shot_id, None)

    def reject_attempt(self, planned: Any, scene_index: int, attempt: int) -> None:
        """Discard an attempt without changing durable scene continuity memory."""
        del scene_index
        shot_id = str(getattr(planned, "shot_id", ""))
        state = self._active.get(shot_id)
        if state is None:
            return
        if int(state.metadata.get("film_attempt", -1)) != attempt:
            raise ValueError("rejected attempt does not match active temporal state")
        self._active.pop(shot_id, None)

    def snapshot(self) -> dict[str, object]:
        """Return durable runtime state; in-flight attempts are intentionally absent."""
        return {
            "kind": FILM_CONTINUITY_RUNTIME_KIND,
            "memory": self.memory.snapshot(),
        }

    def restore(self, payload: dict[str, object]) -> None:
        """Restore durable accepted continuity and clear any in-flight attempt."""
        if str(payload.get("kind", "")) != FILM_CONTINUITY_RUNTIME_KIND:
            raise ValueError("unsupported native film continuity runtime kind")
        raw_memory = payload.get("memory")
        if not isinstance(raw_memory, dict):
            raise ValueError("native film continuity runtime is missing memory")
        self.memory = SceneContinuityMemory.restore(raw_memory)
        self._active.clear()
