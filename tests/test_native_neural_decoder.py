import pytest

from cineos.native_image.neural_backend import (
    NeuralModelConfig,
    _load_torch,
    torch_available,
)
from cineos.native_image.neural_decoder import (
    TorchLatentRGBDecoder,
    save_latent_comparison,
)


def _config():
    return NeuralModelConfig(
        feature_dim=4,
        embedding_dim=8,
        latent_dim=6,
        hidden_dim=12,
    )


def test_neural_decoder_requires_optional_backend():
    if torch_available():
        pytest.skip("dependency contract only applies without PyTorch")
    with pytest.raises(RuntimeError, match=r"cineos\[neural\]"):
        TorchLatentRGBDecoder(_config())


def test_decoder_writes_reconstruction_and_generated_frames(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = _load_torch()
    decoder = TorchLatentRGBDecoder(_config(), width=8, height=6)
    target = torch.zeros(6)
    generated = torch.ones(6) * 0.25
    artifacts = save_latent_comparison(decoder, target, generated, tmp_path)

    reconstruction = tmp_path / "reconstruction.ppm"
    output = tmp_path / "generated.ppm"
    assert artifacts.reconstruction_path == str(reconstruction)
    assert artifacts.generated_path == str(output)
    assert reconstruction.read_bytes().startswith(b"P6\n8 6\n255\n")
    assert output.read_bytes().startswith(b"P6\n8 6\n255\n")
