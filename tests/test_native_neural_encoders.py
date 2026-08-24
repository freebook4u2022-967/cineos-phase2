import pytest

from cineos.native_image.image_pixels import pillow_available
from cineos.native_image.neural_backend import NeuralModelConfig, torch_available
from cineos.native_image.neural_encoders import (
    TorchCharacterReferenceEncoder,
    TorchImageLatentEncoder,
    TorchSceneTextEncoder,
)


def _config() -> NeuralModelConfig:
    return NeuralModelConfig(
        feature_dim=4,
        embedding_dim=8,
        latent_dim=6,
        hidden_dim=12,
        image_size=8,
    )


def test_neural_encoders_require_optional_torch_when_missing(tmp_path):
    if torch_available():
        pytest.skip("PyTorch is available in this environment")
    with pytest.raises(RuntimeError, match=r"cineos\[neural\]"):
        TorchImageLatentEncoder(_config())


def test_neural_encoders_produce_expected_shapes_when_available(tmp_path):
    if not torch_available() or not pillow_available():
        pytest.skip("CINEOS neural optional dependencies are not installed")

    from PIL import Image

    image = tmp_path / "image.png"
    ref_a = tmp_path / "front.png"
    ref_b = tmp_path / "profile.png"
    Image.new("RGB", (12, 10), (180, 80, 40)).save(image)
    Image.new("RGB", (10, 12), (120, 90, 60)).save(ref_a)
    Image.new("RGB", (11, 11), (125, 95, 65)).save(ref_b)

    image_encoder = TorchImageLatentEncoder(_config())
    mean, logvar = image_encoder.encode_file(image)
    latent = image_encoder.sample(mean, logvar, deterministic=True)
    identity = TorchCharacterReferenceEncoder(_config()).encode_files((ref_a, ref_b))
    scene = TorchSceneTextEncoder(_config()).encode(
        "Hero turns",
        "Rainy port",
        ("same wardrobe",),
    )

    assert tuple(mean.shape) == (6,)
    assert tuple(logvar.shape) == (6,)
    assert tuple(latent.shape) == (6,)
    assert tuple(identity.shape) == (4,)
    assert tuple(scene.shape) == (4,)


def test_image_encoder_uses_decoded_pixels_not_container_bytes(tmp_path):
    if not torch_available() or not pillow_available():
        pytest.skip("CINEOS neural optional dependencies are not installed")

    import torch
    from PIL import Image

    pixels = Image.new("RGB", (16, 16), (25, 140, 210))
    low_compression = tmp_path / "low.png"
    high_compression = tmp_path / "high.png"
    pixels.save(low_compression, compress_level=0)
    pixels.save(high_compression, compress_level=9)
    assert low_compression.read_bytes() != high_compression.read_bytes()

    encoder = TorchImageLatentEncoder(_config())
    low_mean, low_logvar = encoder.encode_file(low_compression)
    high_mean, high_logvar = encoder.encode_file(high_compression)

    assert torch.allclose(low_mean, high_mean)
    assert torch.allclose(low_logvar, high_logvar)
