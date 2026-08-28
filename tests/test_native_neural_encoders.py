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


def test_character_reference_encoder_deduplicates_decoded_pixel_evidence(tmp_path):
    if not torch_available() or not pillow_available():
        pytest.skip("CINEOS neural optional dependencies are not installed")

    import torch
    from PIL import Image

    front = Image.new("RGB", (16, 16), (95, 130, 175))
    front_low = tmp_path / "front-low.png"
    front_high = tmp_path / "front-high.png"
    profile = tmp_path / "profile.png"
    front.save(front_low, compress_level=0)
    front.save(front_high, compress_level=9)
    Image.new("RGB", (16, 16), (115, 105, 165)).save(profile)
    assert front_low.read_bytes() != front_high.read_bytes()

    torch.manual_seed(11)
    encoder = TorchCharacterReferenceEncoder(_config())
    unique = encoder.encode_files((front_low, profile))
    duplicated = encoder.encode_files((front_low, front_high, profile))
    reordered = encoder.encode_files((profile, front_high, front_low))

    assert torch.allclose(unique, duplicated)
    assert torch.allclose(unique, reordered)


def test_scene_text_encoder_uses_stable_trainable_token_embeddings():
    if not torch_available():
        pytest.skip("CINEOS neural optional dependencies are not installed")

    import torch

    torch.manual_seed(7)
    encoder = TorchSceneTextEncoder(
        _config(),
        vocabulary_size=257,
        max_tokens=32,
    )

    first_ids = encoder.tokenize("Rainy port, same wardrobe")
    repeated_ids = encoder.tokenize("Rainy port, same wardrobe")
    changed_ids = encoder.tokenize("Sunny desert, torn wardrobe")

    assert first_ids == repeated_ids
    assert first_ids != changed_ids
    assert len(first_ids) <= 32
    assert all(0 <= token_id < 257 for token_id in first_ids)

    rainy = encoder.encode("Hero turns", "Rainy port", ("same wardrobe",))
    sunny = encoder.encode("Hero turns", "Sunny desert", ("same wardrobe",))
    assert tuple(rainy.shape) == (4,)
    assert not torch.allclose(rainy, sunny)

    rainy.sum().backward()
    gradient = encoder.embedding.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0


def test_scene_text_encoder_rejects_invalid_tokenizer_configuration():
    if not torch_available():
        pytest.skip("CINEOS neural optional dependencies are not installed")

    with pytest.raises(ValueError, match="vocabulary_size"):
        TorchSceneTextEncoder(_config(), vocabulary_size=1)
    with pytest.raises(ValueError, match="max_tokens"):
        TorchSceneTextEncoder(_config(), max_tokens=0)
