"""Versioned scene-to-scene continuity memory for CINEOS native video.

A temporal model normally owns recurrent state only inside one shot.  Long-form
film production also needs an explicit boundary that can carry accepted visual
state into the next shot or scene without pretending two shots are one sequence.
This module provides that boundary.  Only committed temporal state may become a
scene anchor; callers can choose a soft continuity transfer or an explicit hard
cut, and the complete memory is JSON-safe for resumable production jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cineos.native_image.tensor_model import Tensor

from .temporal_model import NativeTemporalModel, TemporalSequenceState

SCENE_MEMORY_SCHEMA = "cineos-scene-continuity-memory/0.1"


@dataclass(frozen=True, slots=True)
class SceneTransitionPolicy:
    """Controls how much accepted state crosses a shot/scene boundary."""

    hidden_carry: float = 0.65
    preserve_latent_reference: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.hidden_carry <= 1.0:
            raise ValueError("hidden_carry must be between 0 and 1")

    @classmethod
    def hard_cut(cls) -> SceneTransitionPolicy:
        """Return a policy that intentionally resets temporal visual memory."""
        return cls(hidden_carry=0.0, preserve_latent_reference=False)


@dataclass(frozen=True, slots=True)
class SceneContinuityAnchor:
    """Accepted end-of-shot state eligible to seed a later shot.

    Native artifact provenance is optional for backwards compatibility with
    checkpoints produced before the renderer-integrity boundary existed. New
    native-film renders persist both values together so checkpoint/resume retains
    a cryptographic link between the accepted visual state and its actual file.
    """

    scene_index: int
    shot_id: str
    hidden: Tensor
    latent: Tensor
    frame_index: int
    native_artifact_sha256: str | None = None
    native_artifact_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.scene_index < 0:
            raise ValueError("scene_index must be non-negative")
        if not self.shot_id:
            raise ValueError("continuity anchor requires a shot_id")
        if self.frame_index < 0:
            raise ValueError("continuity anchor requires an accepted frame")
        if self.hidden.device != self.latent.device:
            raise ValueError("anchor hidden and latent tensors must share a device")
        has_digest = self.native_artifact_sha256 is not None
        has_size = self.native_artifact_bytes is not None
        if has_digest != has_size:
            raise ValueError("native artifact provenance must include digest and byte size")
        if has_digest:
            digest = str(self.native_artifact_sha256)
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("native artifact sha256 must be a lowercase hex digest")
            if int(self.native_artifact_bytes) <= 0:
                raise ValueError("native artifact byte size must be positive")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "scene_index": self.scene_index,
            "shot_id": self.shot_id,
            "hidden": list(self.hidden.values),
            "hidden_shape": list(self.hidden.shape),
            "latent": list(self.latent.values),
            "latent_shape": list(self.latent.shape),
            "device": self.hidden.device,
            "frame_index": self.frame_index,
        }
        if self.native_artifact_sha256 is not None:
            payload["native_artifact_sha256"] = self.native_artifact_sha256
            payload["native_artifact_bytes"] = self.native_artifact_bytes
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> SceneContinuityAnchor:
        hidden_values = payload.get("hidden")
        hidden_shape = payload.get("hidden_shape")
        latent_values = payload.get("latent")
        latent_shape = payload.get("latent_shape")
        if not all(
            isinstance(value, list)
            for value in (hidden_values, hidden_shape, latent_values, latent_shape)
        ):
            raise ValueError("scene continuity anchor is missing tensor data")
        raw_digest = payload.get("native_artifact_sha256")
        raw_bytes = payload.get("native_artifact_bytes")
        device = str(payload.get("device", "cpu"))
        return cls(
            scene_index=int(payload.get("scene_index", -1)),
            shot_id=str(payload.get("shot_id", "")),
            hidden=Tensor(
                tuple(float(value) for value in hidden_values),
                tuple(int(value) for value in hidden_shape),
                device,
            ),
            latent=Tensor(
                tuple(float(value) for value in latent_values),
                tuple(int(value) for value in latent_shape),
                device,
            ),
            frame_index=int(payload.get("frame_index", -1)),
            native_artifact_sha256=(
                None if raw_digest is None else str(raw_digest)
            ),
            native_artifact_bytes=(None if raw_bytes is None else int(raw_bytes)),
        )


@dataclass(slots=True)
class SceneContinuityMemory:
    """Append-only accepted visual memory across shots and scenes.

    Recording is intentionally separated from generation: a caller records an
    anchor only after the shot has passed temporal/visual QC.  Starting another
    shot creates a fresh sequence index while optionally carrying a decayed
    hidden state and the previous accepted latent as a continuity reference.
    """

    anchors: list[SceneContinuityAnchor] = field(default_factory=list)
    schema: str = SCENE_MEMORY_SCHEMA

    def latest(self) -> SceneContinuityAnchor | None:
        return self.anchors[-1] if self.anchors else None

    def record_accepted_shot(
        self,
        *,
        scene_index: int,
        state: TemporalSequenceState,
    ) -> SceneContinuityAnchor:
        """Commit an accepted shot end-state to long-range continuity memory."""
        if scene_index < 0:
            raise ValueError("scene_index must be non-negative")
        if state.last_frame_index < 0 or state.last_latent is None:
            raise ValueError("cannot record a shot without an accepted frame")
        previous = self.latest()
        if previous is not None:
            if scene_index < previous.scene_index:
                raise ValueError("scene_index must not move backwards")
            if state.shot_id == previous.shot_id:
                raise ValueError("the same shot_id cannot be recorded twice")

        raw_digest = state.metadata.get("native_artifact_sha256")
        raw_bytes = state.metadata.get("native_artifact_bytes")
        anchor = SceneContinuityAnchor(
            scene_index=scene_index,
            shot_id=state.shot_id,
            hidden=state.hidden,
            latent=state.last_latent,
            frame_index=state.last_frame_index,
            native_artifact_sha256=(
                None if raw_digest is None else str(raw_digest)
            ),
            native_artifact_bytes=(None if raw_bytes is None else int(raw_bytes)),
        )
        self.anchors.append(anchor)
        return anchor

    def start_shot(
        self,
        model: NativeTemporalModel,
        *,
        scene_index: int,
        shot_id: str,
        device: str = "cpu",
        policy: SceneTransitionPolicy | None = None,
    ) -> TemporalSequenceState:
        """Create a fresh shot state seeded from the last accepted anchor."""
        if scene_index < 0:
            raise ValueError("scene_index must be non-negative")
        if not shot_id:
            raise ValueError("shot_id must not be empty")
        previous = self.latest()
        if previous is not None and scene_index < previous.scene_index:
            raise ValueError("scene_index must not move backwards")

        state = model.initial_state(shot_id, device=device)
        if previous is None:
            state.metadata["continuity_source"] = "none"
            state.metadata["scene_index"] = scene_index
            return state

        transition = policy or SceneTransitionPolicy()
        if previous.hidden.shape != state.hidden.shape:
            raise ValueError(
                "continuity anchor hidden shape is incompatible with model"
            )
        if previous.hidden.device != device or previous.latent.device != device:
            raise ValueError("continuity anchor and new shot must share a device")
        if previous.latent.shape != (model.latent_dim,):
            raise ValueError(
                "continuity anchor latent shape is incompatible with model"
            )

        state.hidden = Tensor(
            tuple(value * transition.hidden_carry for value in previous.hidden.values),
            previous.hidden.shape,
            device,
        )
        if transition.preserve_latent_reference:
            state.last_latent = previous.latent
        state.metadata.update(
            {
                "continuity_source": "scene-anchor",
                "scene_index": scene_index,
                "previous_scene_index": previous.scene_index,
                "previous_shot_id": previous.shot_id,
                "hidden_carry": transition.hidden_carry,
                "latent_reference_preserved": int(transition.preserve_latent_reference),
            }
        )
        return state

    def snapshot(self) -> dict[str, object]:
        """Return a stable JSON-safe production checkpoint payload."""
        return {
            "schema": self.schema,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
        }

    @classmethod
    def restore(cls, payload: dict[str, object]) -> SceneContinuityMemory:
        """Restore and validate an append-only continuity memory checkpoint."""
        schema = str(payload.get("schema", ""))
        if schema != SCENE_MEMORY_SCHEMA:
            raise ValueError(f"unsupported scene continuity memory schema: {schema}")
        raw_anchors = payload.get("anchors", [])
        if not isinstance(raw_anchors, list):
            raise ValueError("scene continuity anchors must be a list")

        memory = cls()
        previous_scene = -1
        seen_shots: set[str] = set()
        for item in raw_anchors:
            if not isinstance(item, dict):
                raise ValueError("scene continuity anchor must be a mapping")
            anchor = SceneContinuityAnchor.from_dict(item)
            if anchor.scene_index < previous_scene:
                raise ValueError(
                    "checkpoint scene_index values must not move backwards"
                )
            if anchor.shot_id in seen_shots:
                raise ValueError("checkpoint contains duplicate shot_id values")
            memory.anchors.append(anchor)
            previous_scene = anchor.scene_index
            seen_shots.add(anchor.shot_id)
        return memory
