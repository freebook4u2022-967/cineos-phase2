from __future__ import annotations

import pytest

from cineos.native_image.tensor_model import Tensor
from cineos.native_video.temporal_model import (
    NativeTemporalModel,
    TemporalFrameInput,
    TemporalSequenceState,
)


def _frame(index: int, *, shot_id: str = "shot-001") -> TemporalFrameInput:
    return TemporalFrameInput(
        shot_id=shot_id,
        frame_index=index,
        identity=Tensor((0.1,) * 8, (8,)),
        scene=Tensor((0.2,) * 8, (8,)),
        motion=Tensor((0.05 * (index + 1),) * 4, (4,)),
    )


def test_temporal_model_advances_sequence_and_tracks_continuity() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")

    first = model.step(_frame(0), state)
    second = model.step(_frame(1), state)

    assert first.frame_index == 0
    assert first.continuity_delta == 0.0
    assert second.frame_index == 1
    assert second.continuity_delta >= 0.0
    assert state.last_frame_index == 1
    assert state.last_latent == second.latent
    assert state.metadata["frames_generated"] == 2


def test_temporal_model_rejects_out_of_order_frames() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")

    with pytest.raises(ValueError, match="expected frame_index 0"):
        model.step(_frame(1), state)


def test_temporal_state_snapshot_restore_is_resumable() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    model.step(_frame(0), state)

    restored = TemporalSequenceState.restore(state.snapshot())
    resumed = model.step(_frame(1), restored)

    assert restored.shot_id == "shot-001"
    assert restored.last_frame_index == 1
    assert resumed.frame_index == 1
    assert restored.metadata["frames_generated"] == 2


def test_temporal_model_rejects_cross_shot_state() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")

    with pytest.raises(ValueError, match="same shot"):
        model.step(_frame(0, shot_id="shot-002"), state)
