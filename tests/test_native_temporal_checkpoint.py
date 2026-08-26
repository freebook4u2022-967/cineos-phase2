from __future__ import annotations

from copy import deepcopy

import pytest

from cineos.native_image.tensor_model import Tensor
from cineos.native_video.temporal_checkpoint import (
    TEMPORAL_CHECKPOINT_SCHEMA_VERSION,
    TemporalCheckpointError,
    restore_temporal_checkpoint,
    temporal_checkpoint_payload,
)
from cineos.native_video.temporal_model import NativeTemporalModel, TemporalFrameInput


def _frame(index: int) -> TemporalFrameInput:
    return TemporalFrameInput(
        shot_id="shot-001",
        frame_index=index,
        identity=Tensor((0.1,) * 8, (8,)),
        scene=Tensor((0.2,) * 8, (8,)),
        motion=Tensor((0.05,) * 4, (4,)),
    )


def test_temporal_checkpoint_round_trip_preserves_resumable_state() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    model.step(_frame(0), state)

    document = temporal_checkpoint_payload(state)
    restored = restore_temporal_checkpoint(document)
    resumed = model.step(_frame(1), restored)

    assert document["schema_version"] == TEMPORAL_CHECKPOINT_SCHEMA_VERSION
    assert restored.shot_id == "shot-001"
    assert resumed.frame_index == 1
    assert restored.last_frame_index == 1
    assert restored.metadata["frames_generated"] == 2


def test_temporal_checkpoint_rejects_state_tampering() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    document = temporal_checkpoint_payload(state)

    tampered = deepcopy(document)
    tampered["state"]["shot_id"] = "shot-evil"

    with pytest.raises(TemporalCheckpointError, match="hash mismatch"):
        restore_temporal_checkpoint(tampered)


def test_temporal_checkpoint_rejects_unknown_schema() -> None:
    model = NativeTemporalModel.initialized()
    document = temporal_checkpoint_payload(model.initial_state("shot-001"))
    document["schema_version"] = TEMPORAL_CHECKPOINT_SCHEMA_VERSION + 1

    with pytest.raises(TemporalCheckpointError, match="unsupported temporal checkpoint"):
        restore_temporal_checkpoint(document)


def test_temporal_checkpoint_rejects_missing_integrity_hash() -> None:
    model = NativeTemporalModel.initialized()
    document = temporal_checkpoint_payload(model.initial_state("shot-001"))
    document.pop("state_hash")

    with pytest.raises(TemporalCheckpointError, match="missing state hash"):
        restore_temporal_checkpoint(document)


def test_temporal_checkpoint_rejects_inconsistent_started_state() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    model.step(_frame(0), state)
    document = temporal_checkpoint_payload(state)

    invalid = deepcopy(document)
    invalid["state"]["last_latent"] = None
    invalid["state"]["last_latent_shape"] = None
    # Rebuild through the public writer would re-hash the mutated state, so this
    # test creates an internally valid hash while preserving an invalid semantic
    # state by using an independent checkpoint generated from the mutated state.
    from cineos.native_video.temporal_checkpoint import _canonical_hash

    invalid["state_hash"] = _canonical_hash(invalid["state"])

    with pytest.raises(TemporalCheckpointError, match="must contain a last latent"):
        restore_temporal_checkpoint(invalid)


def test_temporal_checkpoint_rejects_metadata_frame_drift() -> None:
    model = NativeTemporalModel.initialized()
    state = model.initial_state("shot-001")
    model.step(_frame(0), state)
    document = temporal_checkpoint_payload(state)

    invalid = deepcopy(document)
    invalid["state"]["metadata"]["frames_generated"] = 99
    from cineos.native_video.temporal_checkpoint import _canonical_hash

    invalid["state_hash"] = _canonical_hash(invalid["state"])

    with pytest.raises(TemporalCheckpointError, match="does not match"):
        restore_temporal_checkpoint(invalid)
