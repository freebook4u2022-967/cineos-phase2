from __future__ import annotations

import pytest

from cineos.native_image.tensor_model import Tensor
from cineos.native_video import (
    NativeTemporalModel,
    SceneContinuityMemory,
    SceneTransitionPolicy,
)


def _accepted_state(model: NativeTemporalModel, shot_id: str):
    state = model.initial_state(shot_id)
    state.hidden = Tensor(
        tuple(0.5 for _ in range(model.hidden_dim)), (model.hidden_dim,)
    )
    state.last_latent = Tensor(
        tuple(0.25 for _ in range(model.latent_dim)), (model.latent_dim,)
    )
    state.last_frame_index = 11
    state.metadata["accepted_candidates"] = 12
    return state


def test_soft_transition_carries_accepted_visual_state_into_new_shot() -> None:
    model = NativeTemporalModel.initialized()
    memory = SceneContinuityMemory()
    first = _accepted_state(model, "scene-0-shot-2")

    memory.record_accepted_shot(scene_index=0, state=first)
    next_state = memory.start_shot(
        model,
        scene_index=1,
        shot_id="scene-1-shot-0",
        policy=SceneTransitionPolicy(hidden_carry=0.5),
    )

    assert next_state.shot_id == "scene-1-shot-0"
    assert next_state.last_frame_index == -1
    assert next_state.last_latent == first.last_latent
    assert next_state.hidden.values == tuple(0.25 for _ in range(model.hidden_dim))
    assert next_state.metadata["previous_shot_id"] == "scene-0-shot-2"
    assert next_state.metadata["previous_scene_index"] == 0


def test_hard_cut_explicitly_resets_cross_scene_visual_memory() -> None:
    model = NativeTemporalModel.initialized()
    memory = SceneContinuityMemory()
    memory.record_accepted_shot(
        scene_index=0,
        state=_accepted_state(model, "shot-a"),
    )

    state = memory.start_shot(
        model,
        scene_index=1,
        shot_id="shot-b",
        policy=SceneTransitionPolicy.hard_cut(),
    )

    assert state.last_latent is None
    assert set(state.hidden.values) == {0.0}
    assert state.metadata["latent_reference_preserved"] == 0
    assert state.metadata["hidden_carry"] == 0.0


def test_unaccepted_or_duplicate_shots_cannot_poison_memory() -> None:
    model = NativeTemporalModel.initialized()
    memory = SceneContinuityMemory()

    with pytest.raises(ValueError, match="accepted frame"):
        memory.record_accepted_shot(
            scene_index=0,
            state=model.initial_state("unaccepted"),
        )

    accepted = _accepted_state(model, "shot-a")
    memory.record_accepted_shot(scene_index=2, state=accepted)

    with pytest.raises(ValueError, match="same shot_id"):
        memory.record_accepted_shot(scene_index=2, state=accepted)

    with pytest.raises(ValueError, match="move backwards"):
        memory.start_shot(model, scene_index=1, shot_id="shot-b")


def test_scene_continuity_memory_checkpoint_roundtrip() -> None:
    model = NativeTemporalModel.initialized()
    memory = SceneContinuityMemory()
    memory.record_accepted_shot(
        scene_index=3,
        state=_accepted_state(model, "shot-x"),
    )

    restored = SceneContinuityMemory.restore(memory.snapshot())

    assert restored.snapshot() == memory.snapshot()
    resumed = restored.start_shot(model, scene_index=4, shot_id="shot-y")
    assert resumed.metadata["continuity_source"] == "scene-anchor"
    assert resumed.metadata["previous_shot_id"] == "shot-x"


def test_restore_rejects_duplicate_shot_ids() -> None:
    model = NativeTemporalModel.initialized()
    memory = SceneContinuityMemory()
    anchor = memory.record_accepted_shot(
        scene_index=0,
        state=_accepted_state(model, "shot-a"),
    ).to_dict()
    payload = memory.snapshot()
    payload["anchors"] = [anchor, dict(anchor)]

    with pytest.raises(ValueError, match="duplicate shot_id"):
        SceneContinuityMemory.restore(payload)
