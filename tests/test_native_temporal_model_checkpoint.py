from __future__ import annotations

import json

import pytest

from cineos.native_video.temporal_model import NativeTemporalModel
from cineos.native_video.temporal_model_checkpoint import (
    TemporalModelCheckpoint,
    TemporalModelCheckpointError,
)

_DATA_FINGERPRINT = "a" * 64


def _checkpoint() -> TemporalModelCheckpoint:
    return TemporalModelCheckpoint.capture(
        NativeTemporalModel.initialized(),
        training_steps=250,
        training_run_id="temporal-run-001",
        training_data_fingerprint=_DATA_FINGERPRINT,
    )


def test_temporal_model_checkpoint_round_trip(tmp_path) -> None:
    checkpoint = _checkpoint()
    path = checkpoint.save(tmp_path / "temporal.json")

    restored = TemporalModelCheckpoint.load(path)

    assert restored.sha256 == checkpoint.sha256
    assert restored.training_steps == 250
    assert restored.training_run_id == "temporal-run-001"
    assert restored.model.identity_dim == checkpoint.model.identity_dim
    assert restored.model.recurrent.weights == checkpoint.model.recurrent.weights
    assert restored.model.decoder.bias == checkpoint.model.decoder.bias


def test_temporal_model_checkpoint_rejects_bootstrap_claim() -> None:
    with pytest.raises(
        TemporalModelCheckpointError,
        match="positive training_steps",
    ):
        TemporalModelCheckpoint.capture(
            NativeTemporalModel.initialized(),
            training_steps=0,
            training_run_id="bootstrap",
            training_data_fingerprint=_DATA_FINGERPRINT,
        )


def test_temporal_model_checkpoint_rejects_tampered_weights(tmp_path) -> None:
    path = _checkpoint().save(tmp_path / "temporal.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"]["decoder"]["weights"][0] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TemporalModelCheckpointError, match="hash mismatch"):
        TemporalModelCheckpoint.load(path)


def test_temporal_model_checkpoint_rejects_incompatible_layer_shape() -> None:
    payload = _checkpoint().to_dict()
    payload["model"]["recurrent"]["input_dim"] += 1
    payload["model"]["recurrent"]["weights"].extend(
        [0.0] * payload["model"]["recurrent"]["output_dim"]
    )

    with pytest.raises(
        TemporalModelCheckpointError,
        match="recurrent layer shape is incompatible",
    ):
        TemporalModelCheckpoint.from_dict(payload, verify_hash=False)


def test_temporal_model_checkpoint_requires_data_provenance() -> None:
    with pytest.raises(
        TemporalModelCheckpointError,
        match="training_data_fingerprint",
    ):
        TemporalModelCheckpoint.capture(
            NativeTemporalModel.initialized(),
            training_steps=1,
            training_run_id="run",
            training_data_fingerprint="not-a-digest",
        )
