import pytest

from cineos.native_image.neural_backend import (
    NeuralModelConfig,
    _load_torch,
    torch_available,
)
from cineos.native_image.neural_decoder import TorchLatentRGBDecoder
from cineos.native_image.tensor_model import Tensor
from cineos.native_video.learned_decoder import CheckpointLatentRGBDecoder


def _config():
    return NeuralModelConfig(
        feature_dim=4,
        embedding_dim=8,
        latent_dim=6,
        hidden_dim=12,
    )


def _checkpoint(tmp_path):
    decoder = TorchLatentRGBDecoder(_config(), width=8, height=6)
    return decoder.save_checkpoint(tmp_path / "decoder.pt")


def test_checkpoint_decoder_requires_real_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="decoder checkpoint does not exist"):
        CheckpointLatentRGBDecoder(tmp_path / "missing.pt")


def test_checkpoint_decoder_produces_renderer_rgb_bytes(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    checkpoint = _checkpoint(tmp_path)
    decoder = CheckpointLatentRGBDecoder(checkpoint)
    latent = Tensor(tuple(float(index) / 10.0 for index in range(6)), (6,), "cpu")

    rgb = decoder.decode(latent, width=8, height=6)

    assert len(rgb) == 8 * 6 * 3
    assert decoder.latent_dim == 6
    assert decoder.decoder_id.startswith("cineos-torch-rgb-decoder/0.1@sha256:")


def test_checkpoint_decoder_fails_closed_on_latent_mismatch(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    decoder = CheckpointLatentRGBDecoder(_checkpoint(tmp_path))
    latent = Tensor((0.0,) * 5, (5,), "cpu")

    with pytest.raises(ValueError, match="latent shape mismatch"):
        decoder.decode(latent, width=8, height=6)


def test_checkpoint_decoder_fails_closed_on_resolution_mismatch(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    decoder = CheckpointLatentRGBDecoder(_checkpoint(tmp_path))
    latent = Tensor((0.0,) * 6, (6,), "cpu")

    with pytest.raises(ValueError, match="resolution mismatch"):
        decoder.decode(latent, width=16, height=12)


def test_checkpoint_fingerprint_changes_with_weights(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = _load_torch()
    first = TorchLatentRGBDecoder(_config(), width=8, height=6)
    second = TorchLatentRGBDecoder(_config(), width=8, height=6)
    with torch.no_grad():
        for parameter in second.network.parameters():
            parameter.add_(0.125)
            break

    first_path = first.save_checkpoint(tmp_path / "first.pt")
    second_path = second.save_checkpoint(tmp_path / "second.pt")

    first_deployed = CheckpointLatentRGBDecoder(first_path)
    second_deployed = CheckpointLatentRGBDecoder(second_path)
    assert first_deployed.decoder_id != second_deployed.decoder_id
