import pytest

from cineos.native_image.tensor_model import CineosTensorModel, Tensor
from cineos.native_image.tensor_training import (
    FlowMatchingBatch,
    TensorBatchTrainer,
    TensorSGDOptimizer,
    flow_matching_gradients,
    flow_matching_objective,
    move_tensor,
)


def _features(offset: float = 0.0) -> Tensor:
    return Tensor(tuple((index / 10.0) + offset for index in range(8)), (8,))


def _latent(value: float) -> Tensor:
    return Tensor(tuple(value for _ in range(16)), (16,))


def test_flow_matching_objective_returns_velocity_loss():
    model = CineosTensorModel.initialized()
    result = flow_matching_objective(
        model,
        _features(),
        _features(0.2),
        _latent(-0.5),
        _latent(0.5),
        0.25,
    )
    assert result.loss >= 0.0
    assert result.predicted_velocity.shape == (16,)
    assert result.target_velocity.values == pytest.approx((1.0,) * 16)
    assert result.interpolated_latent.values == pytest.approx((-0.25,) * 16)


def test_flow_matching_gradients_cover_all_trainable_layers():
    model = CineosTensorModel.initialized()
    result, identity_gradients, scene_gradients, latent_gradients = (
        flow_matching_gradients(
            model,
            _features(),
            _features(0.2),
            _latent(-0.5),
            _latent(0.5),
            0.25,
        )
    )
    assert result.loss > 0.0
    assert len(identity_gradients.weights) == len(model.identity_encoder.weights)
    assert len(scene_gradients.weights) == len(model.scene_encoder.weights)
    assert len(latent_gradients.weights) == len(model.latent_network.weights)
    assert any(abs(value) > 0.0 for value in latent_gradients.weights)


def test_tensor_batch_training_updates_model_weights():
    model = CineosTensorModel.initialized()
    trainer = TensorBatchTrainer(model, TensorSGDOptimizer(learning_rate=0.01))
    before = tuple(model.latent_network.weights)
    batch = FlowMatchingBatch(
        identity_features=(_features(), _features(0.1)),
        scene_features=(_features(0.2), _features(0.3)),
        source_latents=(_latent(-0.5), _latent(-0.25)),
        target_latents=(_latent(0.5), _latent(0.75)),
        times=(0.25, 0.75),
    )
    loss = trainer.train_batch(batch)
    assert loss >= 0.0
    assert trainer.step == 1
    assert tuple(model.latent_network.weights) != before


def test_repeated_analytic_training_reduces_flow_loss():
    model = CineosTensorModel.initialized()
    trainer = TensorBatchTrainer(model, TensorSGDOptimizer(learning_rate=0.02))
    batch = FlowMatchingBatch(
        identity_features=(_features(),),
        scene_features=(_features(0.2),),
        source_latents=(_latent(-0.5),),
        target_latents=(_latent(0.5),),
        times=(0.25,),
    )
    before = flow_matching_objective(
        model,
        batch.identity_features[0],
        batch.scene_features[0],
        batch.source_latents[0],
        batch.target_latents[0],
        batch.times[0],
    ).loss
    for _ in range(40):
        trainer.train_batch(batch)
    after = flow_matching_objective(
        model,
        batch.identity_features[0],
        batch.scene_features[0],
        batch.source_latents[0],
        batch.target_latents[0],
        batch.times[0],
    ).loss
    assert after < before


def test_logical_device_abstraction_validates_devices():
    tensor = move_tensor(_features(), "cuda")
    assert tensor.device == "cuda"
    with pytest.raises(ValueError, match="unsupported tensor device"):
        move_tensor(_features(), "quantum")
