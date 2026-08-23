import pytest

from cineos.native_image.image_experiment import RealImageExperimentRunner
from cineos.native_image.neural_backend import NeuralModelConfig, torch_available
from cineos.native_image.training import NativeDatasetManifest, NativeTrainingSample


def _manifest(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "references").mkdir()
    (tmp_path / "frames" / "a.ppm").write_bytes(b"frame-a")
    (tmp_path / "frames" / "b.ppm").write_bytes(b"frame-b")
    (tmp_path / "references" / "hero.ppm").write_bytes(b"hero")
    samples = [
        NativeTrainingSample(
            sample_id="a",
            image_path="frames/a.ppm",
            character_reference_paths=("references/hero.ppm",),
            caption="Hero looks left",
            scene_description="Interior",
        ),
        NativeTrainingSample(
            sample_id="b",
            image_path="frames/b.ppm",
            character_reference_paths=("references/hero.ppm",),
            caption="Hero looks right",
            scene_description="Interior",
        ),
    ]
    return NativeDatasetManifest("experiment", "1", samples)


def _config():
    return NeuralModelConfig(
        feature_dim=4,
        embedding_dim=8,
        latent_dim=6,
        hidden_dim=12,
    )


def test_real_image_experiment_requires_neural_backend(tmp_path):
    if torch_available():
        pytest.skip("dependency contract only applies without PyTorch")
    with pytest.raises(RuntimeError, match=r"cineos\[neural\]"):
        RealImageExperimentRunner(tmp_path, _config())


def test_real_image_experiment_trains_validates_and_checkpoints(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    runner = RealImageExperimentRunner(tmp_path, _config())
    result = runner.run(
        _manifest(tmp_path),
        checkpoint_path=tmp_path / "experiment.pt",
        epochs=2,
    )
    assert result.training_loss >= 0.0
    assert result.validation_loss >= 0.0
    assert result.training_steps == 2
    assert (tmp_path / "experiment.pt").is_file()
