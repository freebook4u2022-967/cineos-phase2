import pytest

from cineos.native_image.autoencoder import TorchPixelAutoencoder
from cineos.native_image.neural_backend import torch_available


def _write_ppm(path, width=2, height=2):
    rgb = bytes([0, 64, 128, 255, 128, 64, 32, 64, 96, 128, 160, 192])
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb)
    return path


def test_autoencoder_checkpoint_contract_requires_neural_backend(tmp_path):
    if torch_available():
        pytest.skip("dependency contract only applies without PyTorch")
    with pytest.raises(RuntimeError, match=r"cineos\[neural\]"):
        TorchPixelAutoencoder(width=2, height=2)


def test_autoencoder_checkpoint_reload_and_reconstruction_export(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    source = _write_ppm(tmp_path / "input.ppm")
    model = TorchPixelAutoencoder(
        width=2,
        height=2,
        latent_dim=4,
        hidden_dim=8,
    )
    model.train_ppm(source, steps=2)
    checkpoint = tmp_path / "autoencoder.pt"
    export = model.export_reconstruction(
        source,
        tmp_path / "comparison",
        checkpoint_path=checkpoint,
    )
    restored = TorchPixelAutoencoder.load_checkpoint(checkpoint)

    assert restored.step == model.step
    assert (tmp_path / "comparison" / "original.ppm").is_file()
    assert (tmp_path / "comparison" / "reconstructed.ppm").is_file()
    assert export.metrics.mse >= 0.0
    assert export.metrics.mae >= 0.0
    assert export.metrics.psnr > 0.0
