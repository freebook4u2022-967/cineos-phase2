import pytest

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
    )


def test_neural_encoders_require_optional_torch_when_missing(tmp_path):
    if torch_available():
        pytest.skip("PyTorch is available in this environment")
    with pytest.raises(RuntimeError, match=r"cineos\[neural\]"):
        TorchImageLatentEncoder(_config())


def test_neural_encoders_produce_expected_shapes_when_available(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")

    image = tmp_path / "image.ppm"
    ref_a = tmp_path / "front.ppm"
    ref_b = tmp_path / "profile.ppm"
    image.write_bytes(b"P6 image payload")
    ref_a.write_bytes(b"P6 front payload")
    ref_b.write_bytes(b"P6 profile payload")

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
