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
    """Predicted latent plus the recurrent state used for the next frame."""

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
        """Restore a state produced by :meth:`snapshot`."""
        hidden_values = payload.get("hidden")
        hidden_shape = payload.get("hidden_shape")
        if not isinstance(hidden_values, list) or not isinstance(hidden_shape, list):
            raise ValueError("temporal state payload is missing hidden tensor data")
        device = str(payload.get("device", "cpu"))
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

        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise ValueError("temporal state metadata must be a mapping")
        metadata: dict[str, str | int | float] = {}
        for key, value in raw_metadata.items():
            if not isinstance(key, str) or not isinstance(value, (str, int, float)):
                raise ValueError("temporal state metadata must be JSON scalar values")
            metadata[key] = value

        return cls(
            shot_id=str(payload.get("shot_id", "")),
            hidden=hidden,
            last_frame_index=int(payload.get("last_frame_index", -1)),
            last_latent=last_latent,
            metadata=metadata,
        )


@dataclass(slots=True)
class NativeTemporalModel:
    """First CINEOS-owned recurrent latent video model contract.

    The model fuses identity, scene, motion and previous hidden state. A future
    torch implementation can implement the same step/state contract while using
    attention, diffusion/flow matching or other learned temporal objectives.
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

    def step(
        self,
        frame: TemporalFrameInput,
        state: TemporalSequenceState,
    ) -> TemporalFrameOutput:
        """Advance a sequence exactly one frame while enforcing ordering."""
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

        state.hidden = hidden
        state.last_latent = latent
        state.last_frame_index = frame.frame_index
        state.metadata["frames_generated"] = frame.frame_index + 1

        return TemporalFrameOutput(
            shot_id=frame.shot_id,
            frame_index=frame.frame_index,
            latent=latent,
            hidden=hidden,
            continuity_delta=continuity_delta,
        )

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
