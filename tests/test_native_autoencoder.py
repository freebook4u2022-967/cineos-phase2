import pytest

from cineos.native_image.autoencoder import TorchPixelAutoencoder, load_p6_ppm
from cineos.native_image.neural_backend import torch_available


def _write_ppm(path, width=4, height=4):
    rgb = bytes((index * 17) % 256 for index in range(width * height * 3))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb)
    return path


def test_autoencoder_requires_neural_backend(tmp_path):
    if torch_available():
        pytest.skip("dependency contract only applies without PyTorch")
    with pytest.raises(RuntimeError, match=r"cineos\[neural\]"):
        TorchPixelAutoencoder(4, 4)


def test_p6_loader_returns_normalized_real_rgb_pixels(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    path = _write_ppm(tmp_path / "sample.ppm")
    pixels, width, height = load_p6_ppm(path)
    assert (width, height) == (4, 4)
    assert pixels.shape == (4 * 4 * 3,)
    assert float(pixels.min()) >= 0.0
    assert float(pixels.max()) <= 1.0


def test_autoencoder_backprop_updates_parameters_and_tracks_pixel_loss(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    path = _write_ppm(tmp_path / "sample.ppm")
    model = TorchPixelAutoencoder(
        4,
        4,
        latent_dim=6,
        hidden_dim=16,
        learning_rate=1e-2,
    )
    before = next(model.encoder.parameters()).detach().clone()
    result = model.train_ppm(path, steps=2)
    after = next(model.encoder.parameters()).detach().clone()

    assert result.step == 2
    assert result.total_loss >= 0.0
    assert result.reconstruction_loss >= 0.0
    assert result.kl_loss >= 0.0
    assert not before.equal(after)
