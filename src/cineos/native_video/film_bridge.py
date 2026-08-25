"""Bridge native temporal continuity into complete-film orchestration.

The bridge deliberately commits scene continuity only after whole-shot validation
passes. Each retry starts from the last durable accepted scene anchor, so a render
attempt that later fails film-level QC cannot poison the next shot or a resumed
production checkpoint.

Durable checkpoints also carry a deterministic fingerprint of the exact temporal
model architecture and parameters that produced their recurrent state. Restoring a
checkpoint under different weights fails closed instead of silently mixing visual
state from incompatible model generations. Legacy checkpoints remain readable so
existing work is not stranded, while all newly written checkpoints are protected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifact_integrity import NativeArtifactProvenance, verify_continuity_artifact
from .scene_memory import SceneContinuityMemory, SceneTransitionPolicy
from .temporal_model import NativeTemporalModel, TemporalSequenceState

FILM_CONTINUITY_RUNTIME_KIND = "cineos-native-film-continuity/0.1"
TEMPORAL_MODEL_FINGERPRINT_SCHEMA = "cineos-temporal-model-fingerprint/0.1"


def temporal_model_fingerprint(model: NativeTemporalModel) -> str:
    """Return a stable SHA-256 identity for temporal architecture and parameters.

    The payload intentionally includes dimensions, layer topology, weights and
    biases. Device placement is excluded because moving identical weights between
    CPU and GPU must not invalidate a durable continuity checkpoint.
    """
    payload = {
        "schema": TEMPORAL_MODEL_FINGERPRINT_SCHEMA,
        "dimensions": {
            "identity": model.identity_dim,
            "scene": model.scene_dim,
            "motion": model.motion_dim,
            "hidden": model.hidden_dim,
            "latent": model.latent_dim,
        },
        "recurrent": {
            "input_dim": model.recurrent.input_dim,
            "output_dim": model.recurrent.output_dim,
            "weights": model.recurrent.weights,
            "bias": model.recurrent.bias,
        },
        "decoder": {
            "input_dim": model.decoder.input_dim,
            "output_dim": model.decoder.output_dim,
            "weights": model.decoder.weights,
            "bias": model.decoder.bias,
        },
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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

    def orchestrator_kwargs(self) -> dict[str, Any]:
        """Return the complete hook bundle expected by ``FilmOrchestrator``.

        Keeping this binding in the native-video layer avoids model-specific
        imports in the film layer while making the production integration a single
        explicit operation rather than a fragile collection of ad-hoc callbacks.
        """
        return {
            "checkpoint_state_provider": self.snapshot,
            "checkpoint_state_restorer": self.restore,
            "shot_attempt_start": self.start_attempt,
            "shot_attempt_accepted": self.accept_attempt,
            "shot_attempt_rejected": self.reject_attempt,
        }

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
        default = SceneTransitionPolicy()
        return SceneTransitionPolicy(
            hidden_carry=(
                default.hidden_carry if hidden_carry is None else float(hidden_carry)
            ),
            preserve_latent_reference=(
                default.preserve_latent_reference
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

    def verify_latest_artifact(
        self,
        path: str | Path,
        *,
        require_provenance: bool = True,
    ) -> NativeArtifactProvenance | None:
        """Verify a resumed/reused artifact against the latest durable scene anchor.

        Production resume should use the default fail-closed policy. Legacy
        checkpoint migration may explicitly set ``require_provenance=False`` to
        acknowledge that an older anchor cannot be cryptographically verified.
        """
        anchor = self.memory.latest()
        if anchor is None:
            raise ValueError(
                "cannot verify native artifact without a continuity anchor"
            )
        return verify_continuity_artifact(
            anchor,
            path,
            require_provenance=require_provenance,
        )

    def snapshot(self) -> dict[str, object]:
        """Return durable runtime state; in-flight attempts are intentionally absent."""
        return {
            "kind": FILM_CONTINUITY_RUNTIME_KIND,
            "temporal_model_fingerprint": temporal_model_fingerprint(self.model),
            "memory": self.memory.snapshot(),
        }

    def restore(self, payload: dict[str, object]) -> None:
        """Restore durable accepted continuity and clear any in-flight attempt.

        New checkpoints are model-bound. A fingerprint mismatch means recurrent
        state was produced by different temporal weights and is therefore unsafe to
        reuse. Checkpoints written before fingerprinting existed are still accepted
        for backwards compatibility and will become protected on their next save.
        """
        if str(payload.get("kind", "")) != FILM_CONTINUITY_RUNTIME_KIND:
            raise ValueError("unsupported native film continuity runtime kind")
        raw_fingerprint = payload.get("temporal_model_fingerprint")
        if raw_fingerprint is not None:
            expected = temporal_model_fingerprint(self.model)
            if str(raw_fingerprint) != expected:
                raise ValueError(
                    "native film continuity checkpoint temporal model fingerprint "
                    "does not match the active model"
                )
        raw_memory = payload.get("memory")
        if not isinstance(raw_memory, dict):
            raise ValueError("native film continuity runtime is missing memory")
        self.memory = SceneContinuityMemory.restore(raw_memory)
        self._active.clear()
