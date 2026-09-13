import json

import pytest

from cineos.native_image.tensor_checkpoint import (
    TENSOR_CHECKPOINT_SCHEMA,
    TensorCheckpointError,
    TensorTrainingCheckpoint,
)
from cineos.native_image.tensor_model import CineosTensorModel, Tensor
from cineos.native_image.tensor_training import (
    FlowMatchingBatch,
    TensorBatchTrainer,
    TensorSGDOptimizer,
)


def _features(offset: float = 0.0) -> Tensor:
    return Tensor(tuple((index / 10.0) + offset for index in range(8)), (8,))


def _latent(value: float) -> Tensor:
    return Tensor((value,) * 16, (16,))


def _batch() -> FlowMatchingBatch:
    return FlowMatchingBatch(
        identity_features=(_features(),),
        scene_features=(_features(0.2),),
        source_latents=(_latent(-0.5),),
        target_latents=(_latent(0.5),),
        times=(0.25,),
    )


def test_tensor_checkpoint_roundtrip_preserves_training_state(tmp_path):
    trainer = TensorBatchTrainer(
        CineosTensorModel.initialized(),
        TensorSGDOptimizer(learning_rate=0.02),
    )
    trainer.train_batch(_batch())
    trainer.train_batch(_batch())

    checkpoint = TensorTrainingCheckpoint.capture(trainer)
    path = checkpoint.save(tmp_path / "tensor-checkpoint.json")
    restored_checkpoint = TensorTrainingCheckpoint.load(path)
    restored = restored_checkpoint.restore_trainer()

    assert restored_checkpoint.schema == TENSOR_CHECKPOINT_SCHEMA
    assert restored.step == 2
    assert restored.optimizer.learning_rate == pytest.approx(0.02)
    assert (
        restored.model.identity_encoder.weights
        == trainer.model.identity_encoder.weights
    )
    assert restored.model.scene_encoder.weights == trainer.model.scene_encoder.weights
    assert restored.model.latent_network.weights == trainer.model.latent_network.weights


def test_restored_tensor_trainer_continues_training(tmp_path):
    trainer = TensorBatchTrainer(
        CineosTensorModel.initialized(),
        TensorSGDOptimizer(learning_rate=0.01),
    )
    trainer.train_batch(_batch())
    path = TensorTrainingCheckpoint.capture(trainer).save(tmp_path / "checkpoint.json")

    restored = TensorTrainingCheckpoint.load(path).restore_trainer()
    before = tuple(restored.model.latent_network.weights)
    restored.train_batch(_batch())

    assert restored.step == 2
    assert tuple(restored.model.latent_network.weights) != before


def test_tensor_checkpoint_rejects_tampering(tmp_path):
    checkpoint = TensorTrainingCheckpoint.capture(
        TensorBatchTrainer(
            CineosTensorModel.initialized(),
            TensorSGDOptimizer(learning_rate=0.01),
        )
    )
    path = checkpoint.save(tmp_path / "checkpoint.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"]["latent_network"]["weights"][0] += 1.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(TensorCheckpointError, match="hash mismatch"):
        TensorTrainingCheckpoint.load(path)


def test_tensor_checkpoint_rejects_invalid_layer_shape():
    checkpoint = TensorTrainingCheckpoint.capture(
        TensorBatchTrainer(
            CineosTensorModel.initialized(),
            TensorSGDOptimizer(learning_rate=0.01),
        )
    )
    payload = checkpoint.to_dict()
    payload["model"]["identity_encoder"]["weights"].pop()

    with pytest.raises(TensorCheckpointError, match="weight shape mismatch"):
        TensorTrainingCheckpoint.from_dict(payload, verify_hash=False)
