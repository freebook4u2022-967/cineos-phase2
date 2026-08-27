from __future__ import annotations

import math

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
    assert state.metadata["accepted_candidates"] == 2


def test_temporal_model_rejects_out_of_order_frames() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")

    with pytest.raises(ValueError, match="expected frame_index 0"):
        model.step(_frame(1), state)


def test_temporal_state_snapshot_restore_is_resumable() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    model.step(_frame(0), state)

    restored = model.restore_state(state.snapshot())
    resumed = model.step(_frame(1), restored)

    assert restored.shot_id == "shot-001"
    assert restored.last_frame_index == 1
    assert resumed.frame_index == 1
    assert restored.metadata["frames_generated"] == 2
    assert restored.metadata["accepted_candidates"] == 2


def test_temporal_model_rejects_cross_shot_state() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")

    with pytest.raises(ValueError, match="same shot"):
        model.step(_frame(0, shot_id="shot-002"), state)


def test_rejected_candidate_does_not_advance_temporal_state() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    before = state.snapshot()

    rejected = model.propose(_frame(0), state)

    assert rejected.frame_index == 0
    assert state.snapshot() == before
    assert state.last_frame_index == -1
    assert "frames_generated" not in state.metadata


def test_candidate_is_committed_only_after_qc_acceptance() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")

    first_candidate = model.propose(_frame(0), state)
    retry_candidate = model.propose(_frame(0), state)

    assert first_candidate == retry_candidate
    assert state.last_frame_index == -1

    model.commit(retry_candidate, state)

    assert state.last_frame_index == 0
    assert state.last_latent == retry_candidate.latent
    assert state.metadata["frames_generated"] == 1
    assert state.metadata["accepted_candidates"] == 1


def test_stale_candidate_cannot_overwrite_accepted_state() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")

    stale = model.propose(_frame(0), state)
    accepted = model.propose(_frame(0), state)
    model.commit(accepted, state)

    with pytest.raises(ValueError, match="expected candidate frame_index 1"):
        model.commit(stale, state)


def test_temporal_restore_rejects_missing_shot_identity() -> None:
    model = NativeTemporalModel.initialized()
    payload = model.initial_state("shot-001").snapshot()
    payload["shot_id"] = ""

    with pytest.raises(ValueError, match="non-empty shot_id"):
        TemporalSequenceState.restore(payload)


def test_temporal_restore_rejects_incoherent_frame_and_latent_state() -> None:
    model = NativeTemporalModel.initialized()
    payload = model.initial_state("shot-001").snapshot()
    payload["last_frame_index"] = 0

    with pytest.raises(ValueError, match="advanced temporal state requires"):
        TemporalSequenceState.restore(payload)


def test_model_restore_rejects_checkpoint_from_incompatible_model() -> None:
    source = NativeTemporalModel.initialized(hidden_dim=8, latent_dim=6)
    target = NativeTemporalModel.initialized(hidden_dim=16, latent_dim=16)
    state = source.initial_state("shot-001")

    with pytest.raises(ValueError, match="hidden tensor is model-incompatible"):
        target.restore_state(state.snapshot())


def test_model_restore_rejects_non_finite_temporal_state() -> None:
    model = NativeTemporalModel.initialized()
    payload = model.initial_state("shot-001").snapshot()
    payload["hidden"] = [math.nan] + [0.0] * (model.hidden_dim - 1)

    with pytest.raises(ValueError, match="non-finite"):
        model.restore_state(payload)
