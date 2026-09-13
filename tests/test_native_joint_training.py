import pytest

from cineos.native_image.autoencoder import TorchPixelAutoencoder
from cineos.native_image.joint_training import JointConditionalImageTrainer
from cineos.native_image.neural_backend import NeuralModelConfig, torch_available
from cineos.native_image.training import NativeTrainingSample


def _ppm(width: int, height: int, value: int) -> bytes:
    rgb = bytes([value, 255 - value, value // 2] * (width * height))
    return f"P6\n{width} {height}\n255\n".encode("ascii") + rgb


def _config():
    return NeuralModelConfig(
        feature_dim=4,
        embedding_dim=8,
        latent_dim=6,
        hidden_dim=12,
    )


def test_joint_trainer_requires_neural_backend(tmp_path):
    if torch_available():
        pytest.skip("dependency contract only applies without PyTorch")
    with pytest.raises(RuntimeError, match=r"cineos\[neural\]"):
        autoencoder = TorchPixelAutoencoder(4, 4, latent_dim=6, hidden_dim=12)
        JointConditionalImageTrainer(tmp_path, autoencoder, _config())


def test_joint_training_updates_conditional_model(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    (tmp_path / "frames").mkdir()
    (tmp_path / "references").mkdir()
    (tmp_path / "frames" / "shot.ppm").write_bytes(_ppm(4, 4, 64))
    (tmp_path / "references" / "hero.ppm").write_bytes(_ppm(4, 4, 96))
    sample = NativeTrainingSample(
        sample_id="joint-001",
        image_path="frames/shot.ppm",
        character_reference_paths=("references/hero.ppm",),
        caption="Hero turns toward camera",
        scene_description="Night interior",
        continuity_tags=("same-face",),
    )
    autoencoder = TorchPixelAutoencoder(4, 4, latent_dim=6, hidden_dim=12)
    trainer = JointConditionalImageTrainer(tmp_path, autoencoder, _config())
    before = next(trainer.flow_model.parameters()).detach().clone()
    result = trainer.train_sample(sample)
    after = next(trainer.flow_model.parameters()).detach().clone()

    assert result.total_loss >= 0.0
    assert result.reconstruction_loss >= 0.0
    assert result.flow_loss >= 0.0
    assert result.step == 1
    assert not before.equal(after)
