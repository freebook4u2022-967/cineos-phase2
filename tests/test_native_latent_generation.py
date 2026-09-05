import pytest

from cineos.native_image.autoencoder import TorchPixelAutoencoder
from cineos.native_image.latent_generation import ConditionalLatentGenerator
from cineos.native_image.neural_backend import NeuralModelConfig, torch_available


def _config():
    return NeuralModelConfig(
        feature_dim=4,
        embedding_dim=8,
        latent_dim=6,
        hidden_dim=12,
    )


def test_conditional_generator_requires_neural_backend(tmp_path):
    if torch_available():
        pytest.skip("dependency contract only applies without PyTorch")
    with pytest.raises(RuntimeError, match=r"cineos\[neural\]"):
        autoencoder = TorchPixelAutoencoder(4, 4, latent_dim=6, hidden_dim=12)
        ConditionalLatentGenerator(autoencoder, _config())


def test_conditional_generator_produces_pixel_vector(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    reference = tmp_path / "hero.ppm"
    reference.write_bytes(b"P6\n1 1\n255\n\x10\x20\x30")
    autoencoder = TorchPixelAutoencoder(
        4,
        4,
        latent_dim=6,
        hidden_dim=12,
    )
    generator = ConditionalLatentGenerator(
        autoencoder,
        _config(),
        integration_steps=2,
    )
    result = generator.generate(
        (reference,),
        "Hero turns toward camera",
        "Night interior",
        ("same wardrobe",),
        seed=7,
    )
    assert result.latent.shape == (6,)
    assert result.pixels.shape == (4 * 4 * 3,)
