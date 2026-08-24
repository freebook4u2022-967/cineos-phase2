import pytest

from cineos.native_image.identity_bank import CharacterIdentityEmbeddingBank


def test_identity_bank_is_available_without_neural_backend():
    """Identity metadata/QC stays usable in lightweight installations."""
    bank = CharacterIdentityEmbeddingBank()
    entry = bank.build_character("arif", ((1.0, 0.0),))
    assert entry.character_id == "arif"
    assert entry.vector == pytest.approx((1.0, 0.0))


def test_identity_bank_builds_normalized_centroid():
    bank = CharacterIdentityEmbeddingBank()
    entry = bank.build_character("arif", ((1.0, 0.0), (0.8, 0.2), (0.9, 0.1)))
    squared = sum(value * value for value in entry.vector)
    assert squared == pytest.approx(1.0, rel=1e-5)
    assert entry.reference_count == 3
    assert bank.characters() == ("arif",)


def test_identity_similarity_is_high_for_matching_direction():
    bank = CharacterIdentityEmbeddingBank()
    bank.build_character("hana", ((0.0, 1.0), (0.1, 0.9)))
    assert bank.similarity("hana", (0.0, 2.0)) > 0.99


def test_identity_bank_rejects_dimension_mismatch():
    bank = CharacterIdentityEmbeddingBank()
    bank.build_character("arif", ((1.0, 0.0),))
    with pytest.raises(ValueError, match="dimension mismatch"):
        bank.similarity("arif", (1.0, 0.0, 0.0))
