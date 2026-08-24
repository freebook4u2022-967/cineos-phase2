from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.native_image.tensor_model import Tensor
from cineos.native_video import NativeFilmContinuityBridge, NativeTemporalModel


def _planned(shot_id: str, **payload):
    return SimpleNamespace(shot_id=shot_id, payload=payload)


def _accept_one_frame(bridge: NativeFilmContinuityBridge, shot_id: str) -> None:
    state = bridge.state_for(shot_id)
    state.hidden = Tensor((1.0,) * bridge.model.hidden_dim, (bridge.model.hidden_dim,))
    state.last_latent = Tensor(
        (2.0,) * bridge.model.latent_dim, (bridge.model.latent_dim,)
    )
    state.last_frame_index = 0


def test_rejected_whole_shot_retry_does_not_poison_scene_memory():
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    first = _planned("shot-1")
    bridge.start_attempt(first, 0, 1)
    _accept_one_frame(bridge, "shot-1")
    bridge.accept_attempt(first, 0, 1)

    assert len(bridge.memory.anchors) == 1
    durable = bridge.memory.latest()
    assert durable is not None

    second = _planned("shot-2")
    bridge.start_attempt(second, 0, 1)
    rejected = bridge.state_for("shot-2")
    assert rejected.metadata["previous_shot_id"] == "shot-1"
    rejected.hidden = Tensor(
        (99.0,) * bridge.model.hidden_dim, (bridge.model.hidden_dim,)
    )
    rejected.last_latent = Tensor(
        (99.0,) * bridge.model.latent_dim, (bridge.model.latent_dim,)
    )
    rejected.last_frame_index = 0
    bridge.reject_attempt(second, 0, 1)

    assert len(bridge.memory.anchors) == 1
    assert bridge.memory.latest() == durable

    bridge.start_attempt(second, 0, 2)
    retry = bridge.state_for("shot-2")
    assert retry.metadata["film_attempt"] == 2
    assert retry.metadata["previous_shot_id"] == "shot-1"
    assert retry.hidden.values == pytest.approx((0.65,) * bridge.model.hidden_dim)
    assert retry.last_latent == durable.latent


def test_hard_cut_resets_cross_scene_state():
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    first = _planned("shot-1")
    bridge.start_attempt(first, 0, 1)
    _accept_one_frame(bridge, "shot-1")
    bridge.accept_attempt(first, 0, 1)

    second = _planned("shot-2", hard_cut=True)
    bridge.start_attempt(second, 1, 1)
    state = bridge.state_for("shot-2")

    assert state.hidden.values == (0.0,) * bridge.model.hidden_dim
    assert state.last_latent is None
    assert state.metadata["latent_reference_preserved"] == 0


def test_snapshot_restores_only_durable_accepted_state():
    bridge = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    first = _planned("shot-1")
    bridge.start_attempt(first, 0, 1)
    _accept_one_frame(bridge, "shot-1")
    bridge.accept_attempt(first, 0, 1)

    bridge.start_attempt(_planned("shot-in-flight"), 1, 1)
    payload = bridge.snapshot()

    restored = NativeFilmContinuityBridge(model=NativeTemporalModel.initialized())
    restored.restore(payload)

    assert len(restored.memory.anchors) == 1
    assert restored.memory.latest() is not None
    assert restored.memory.latest().shot_id == "shot-1"
    with pytest.raises(KeyError, match="no active temporal state"):
        restored.state_for("shot-in-flight")
