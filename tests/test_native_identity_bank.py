import pytest

from cineos.native_image.identity_bank import CharacterIdentityEmbeddingBank
from cineos.native_image.neural_backend import torch_available


def test_identity_bank_requires_neural_backend():
    if torch_available():
        pytest.skip("dependency contract applies when torch is unavailable")
    with pytest.raises(RuntimeError, match=r"cineos\[neural\]"):
        CharacterIdentityEmbeddingBank().build_character("arif", ((1.0, 0.0),))


def test_identity_bank_builds_normalized_centroid():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    bank = CharacterIdentityEmbeddingBank()
    entry = bank.build_character("arif", ((1.0, 0.0), (0.8, 0.2), (0.9, 0.1)))
    squared = sum(value * value for value in entry.vector)
    assert squared == pytest.approx(1.0, rel=1e-5)
    assert entry.reference_count == 3
    assert bank.characters() == ("arif",)


def test_identity_similarity_is_high_for_matching_direction():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    bank = CharacterIdentityEmbeddingBank()
    bank.build_character("hana", ((0.0, 1.0), (0.1, 0.9)))
    assert bank.similarity("hana", (0.0, 2.0)) > 0.99


def test_identity_bank_rejects_dimension_mismatch():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    bank = CharacterIdentityEmbeddingBank()
    bank.build_character("arif", ((1.0, 0.0),))
    with pytest.raises(ValueError, match="dimension mismatch"):
        bank.similarity("arif", (1.0, 0.0, 0.0))
