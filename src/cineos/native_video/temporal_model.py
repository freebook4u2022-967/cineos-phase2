"""Provider-neutral temporal model foundation for native CINEOS video.

This module intentionally starts with a dependency-light recurrent latent model so
sequence semantics, state persistence and QC integration can stabilize before a
GPU implementation replaces the numerical backend. It does not claim production
video quality; it defines owned temporal behavior that future neural backends must
preserve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from cineos.native_image.tensor_model import LinearTensorLayer, Tensor


@dataclass(frozen=True, slots=True)
class TemporalFrameInput:
    """One frame-step request in a native video sequence."""

    shot_id: str
    frame_index: int
    identity: Tensor
    scene: Tensor
    motion: Tensor

    def __post_init__(self) -> None:
        if not self.shot_id:
            raise ValueError("temporal frame input requires a shot_id")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        devices = {self.identity.device, self.scene.device, self.motion.device}
        if len(devices) != 1:
            raise ValueError("temporal frame tensors must share a device")


@dataclass(frozen=True, slots=True)
class TemporalFrameOutput:
    """Predicted latent plus the recurrent state used for the next frame.

    Outputs are immutable candidates. They are safe to inspect with visual or
    continuity QC before committing them to :class:`TemporalSequenceState`.
    """

    shot_id: str
    frame_index: int
    latent: Tensor
    hidden: Tensor
    continuity_delta: float


@dataclass(slots=True)
class TemporalSequenceState:
    """Resumable temporal state for one shot sequence."""

    shot_id: str
    hidden: Tensor
    last_frame_index: int = -1
    last_latent: Tensor | None = None
    metadata: dict[str, str | int | float] = field(default_factory=dict)

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-safe checkpoint payload for interruption recovery."""
        return {
            "shot_id": self.shot_id,
            "hidden": list(self.hidden.values),
            "hidden_shape": list(self.hidden.shape),
            "device": self.hidden.device,
            "last_frame_index": self.last_frame_index,
            "last_latent": (
                None if self.last_latent is None else list(self.last_latent.values)
            ),
            "last_latent_shape": (
                None if self.last_latent is None else list(self.last_latent.shape)
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def restore(cls, payload: dict[str, object]) -> TemporalSequenceState:
        """Restore and structurally validate a state produced by :meth:`snapshot`.

        Model-specific dimensions are validated by
        :meth:`NativeTemporalModel.restore_state`; this method guarantees the
        serialized state itself is internally coherent before any generation
        resumes.
        """
        shot_id = payload.get("shot_id")
        if not isinstance(shot_id, str) or not shot_id.strip():
            raise ValueError("temporal state payload requires a non-empty shot_id")

        hidden_values = payload.get("hidden")
        hidden_shape = payload.get("hidden_shape")
        if not isinstance(hidden_values, list) or not isinstance(hidden_shape, list):
            raise ValueError("temporal state payload is missing hidden tensor data")
        device_value = payload.get("device", "cpu")
        if not isinstance(device_value, str) or not device_value.strip():
            raise ValueError("temporal state payload requires a non-empty device")
        device = device_value
        hidden = Tensor(
            tuple(float(value) for value in hidden_values),
            tuple(int(value) for value in hidden_shape),
            device,
        )

        latent_values = payload.get("last_latent")
        latent_shape = payload.get("last_latent_shape")
        last_latent: Tensor | None = None
        if latent_values is not None or latent_shape is not None:
            if not isinstance(latent_values, list) or not isinstance(
                latent_shape, list
            ):
                raise ValueError("temporal state payload has incomplete latent data")
            last_latent = Tensor(
                tuple(float(value) for value in latent_values),
                tuple(int(value) for value in latent_shape),
                device,
            )

        raw_last_frame_index = payload.get("last_frame_index", -1)
        if isinstance(raw_last_frame_index, bool) or not isinstance(
            raw_last_frame_index, int
        ):
            raise ValueError("temporal state last_frame_index must be an integer")
        if raw_last_frame_index < -1:
            raise ValueError("temporal state last_frame_index cannot be less than -1")
        if raw_last_frame_index == -1 and last_latent is not None:
            raise ValueError("unstarted temporal state cannot contain a last latent")
        if raw_last_frame_index >= 0 and last_latent is None:
            raise ValueError("advanced temporal state must contain a last latent")

        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise ValueError("temporal state metadata must be a mapping")
        metadata: dict[str, str | int | float] = {}
        for key, value in raw_metadata.items():
            if (
                not isinstance(key, str)
                or isinstance(value, bool)
                or not isinstance(value, (str, int, float))
            ):
                raise ValueError("temporal state metadata must be JSON scalar values")
            metadata[key] = value

        return cls(
            shot_id=shot_id,
            hidden=hidden,
            last_frame_index=raw_last_frame_index,
            last_latent=last_latent,
            metadata=metadata,
        )


@dataclass(slots=True)
class NativeTemporalModel:
    """First CINEOS-owned recurrent latent video model contract.

    The model fuses identity, scene, motion and previous hidden state. A future
    torch implementation can implement the same proposal/commit state contract
    while using attention, diffusion/flow matching or other learned temporal
    objectives.

    Candidate generation is transactional: :meth:`propose` never advances the
    sequence, allowing QC and rerender logic to reject a candidate without
    poisoning continuity memory. :meth:`commit` advances state only after the
    caller accepts the candidate. :meth:`step` preserves the original eager API
    by proposing and committing in one call.
    """

    identity_dim: int
    scene_dim: int
    motion_dim: int
    hidden_dim: int
    latent_dim: int
    recurrent: LinearTensorLayer
    decoder: LinearTensorLayer

    @classmethod
    def initialized(
        cls,
        *,
        identity_dim: int = 8,
        scene_dim: int = 8,
        motion_dim: int = 4,
        hidden_dim: int = 16,
        latent_dim: int = 16,
    ) -> NativeTemporalModel:
        if min(identity_dim, scene_dim, motion_dim, hidden_dim, latent_dim) <= 0:
            raise ValueError("temporal model dimensions must be positive")
        recurrent_input = identity_dim + scene_dim + motion_dim + hidden_dim
        return cls(
            identity_dim=identity_dim,
            scene_dim=scene_dim,
            motion_dim=motion_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            recurrent=LinearTensorLayer.initialized(recurrent_input, hidden_dim),
            decoder=LinearTensorLayer.initialized(hidden_dim, latent_dim),
        )

    def initial_state(
        self, shot_id: str, *, device: str = "cpu"
    ) -> TemporalSequenceState:
        if not shot_id:
            raise ValueError("temporal sequence requires a shot_id")
        return TemporalSequenceState(
            shot_id=shot_id,
            hidden=Tensor((0.0,) * self.hidden_dim, (self.hidden_dim,), device),
        )

    def restore_state(self, payload: dict[str, object]) -> TemporalSequenceState:
        """Restore a checkpoint and enforce this model's dimensional contract.

        Resume should fail at the checkpoint boundary, not several frames later.
        This prevents a stale or incompatible checkpoint from entering a costly
        long-running render job after model upgrades.
        """
        state = TemporalSequenceState.restore(payload)
        if state.hidden.shape != (self.hidden_dim,):
            raise ValueError("temporal checkpoint hidden tensor is model-incompatible")
        if state.last_latent is not None and state.last_latent.shape != (
            self.latent_dim,
        ):
            raise ValueError("temporal checkpoint latent tensor is model-incompatible")
        if any(not math.isfinite(value) for value in state.hidden.values):
            raise ValueError(
                "temporal checkpoint hidden tensor contains non-finite data"
            )
        if state.last_latent is not None and any(
            not math.isfinite(value) for value in state.last_latent.values
        ):
            raise ValueError(
                "temporal checkpoint latent tensor contains non-finite data"
            )
        return state

    def propose(
        self,
        frame: TemporalFrameInput,
        state: TemporalSequenceState,
    ) -> TemporalFrameOutput:
        """Generate the next candidate without mutating sequence state.

        This is the production-safe boundary for QC/retry integration. A caller
        may inspect or reject the returned candidate and call :meth:`propose`
        again with the same state. Only :meth:`commit` advances continuity.
        """
        self._validate_next_frame(frame, state)

        fused = Tensor(
            frame.identity.values
            + frame.scene.values
            + frame.motion.values
            + state.hidden.values,
            (self.identity_dim + self.scene_dim + self.motion_dim + self.hidden_dim,),
            frame.identity.device,
        )
        hidden = self.recurrent.forward(fused)
        latent = self.decoder.forward(hidden)
        continuity_delta = self._continuity_delta(state.last_latent, latent)

        return TemporalFrameOutput(
            shot_id=frame.shot_id,
            frame_index=frame.frame_index,
            latent=latent,
            hidden=hidden,
            continuity_delta=continuity_delta,
        )

    def commit(
        self,
        candidate: TemporalFrameOutput,
        state: TemporalSequenceState,
    ) -> None:
        """Atomically accept a previously proposed candidate into sequence state."""
        if candidate.shot_id != state.shot_id:
            raise ValueError(
                "candidate and temporal state must belong to the same shot"
            )
        expected_index = state.last_frame_index + 1
        if candidate.frame_index != expected_index:
            raise ValueError(
                f"expected candidate frame_index {expected_index}, "
                f"got {candidate.frame_index}"
            )
        if candidate.hidden.shape != (self.hidden_dim,):
            raise ValueError("candidate hidden tensor has incompatible shape")
        if candidate.latent.shape != (self.latent_dim,):
            raise ValueError("candidate latent tensor has incompatible shape")
        if candidate.hidden.device != state.hidden.device:
            raise ValueError("candidate and temporal state must share a device")
        if candidate.latent.device != state.hidden.device:
            raise ValueError("candidate and temporal state must share a device")

        state.hidden = candidate.hidden
        state.last_latent = candidate.latent
        state.last_frame_index = candidate.frame_index
        state.metadata["frames_generated"] = candidate.frame_index + 1
        state.metadata["accepted_candidates"] = (
            int(state.metadata.get("accepted_candidates", 0)) + 1
        )

    def step(
        self,
        frame: TemporalFrameInput,
        state: TemporalSequenceState,
    ) -> TemporalFrameOutput:
        """Advance exactly one accepted frame while preserving the eager API."""
        candidate = self.propose(frame, state)
        self.commit(candidate, state)
        return candidate

    def _validate_next_frame(
        self,
        frame: TemporalFrameInput,
        state: TemporalSequenceState,
    ) -> None:
        if frame.shot_id != state.shot_id:
            raise ValueError("frame and temporal state must belong to the same shot")
        expected_index = state.last_frame_index + 1
        if frame.frame_index != expected_index:
            raise ValueError(
                f"expected frame_index {expected_index}, got {frame.frame_index}"
            )
        if frame.identity.shape != (self.identity_dim,):
            raise ValueError("identity tensor has incompatible shape")
        if frame.scene.shape != (self.scene_dim,):
            raise ValueError("scene tensor has incompatible shape")
        if frame.motion.shape != (self.motion_dim,):
            raise ValueError("motion tensor has incompatible shape")
        if state.hidden.shape != (self.hidden_dim,):
            raise ValueError("temporal hidden state has incompatible shape")
        if frame.identity.device != state.hidden.device:
            raise ValueError("frame tensors and temporal state must share a device")

    @staticmethod
    def _continuity_delta(previous: Tensor | None, current: Tensor) -> float:
        if previous is None:
            return 0.0
        if previous.shape != current.shape:
            raise ValueError("consecutive temporal latents must have identical shapes")
        squared = sum(
            (left - right) ** 2
            for left, right in zip(previous.values, current.values, strict=True)
        ) / len(current.values)
        return math.sqrt(squared)
