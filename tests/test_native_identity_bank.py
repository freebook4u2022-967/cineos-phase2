import math

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


def test_identity_bank_rejects_non_finite_embedding_values():
    bank = CharacterIdentityEmbeddingBank()
    with pytest.raises(ValueError, match="finite values"):
        bank.build_character("arif", ((1.0, math.nan),))
    with pytest.raises(ValueError, match="finite values"):
        bank.build_character("arif", ((math.inf, 1.0),))


def test_robust_identity_fusion_rejects_inconsistent_reference():
    bank = CharacterIdentityEmbeddingBank()
    entry = bank.build_character_robust(
        "arif",
        (
            (1.0, 0.00),
            (0.99, 0.05),
            (0.98, -0.04),
            (-1.0, 0.00),
        ),
        min_consensus_similarity=0.30,
        minimum_references=2,
    )

    assert entry.reference_count == 3
    assert entry.vector[0] > 0.99
    assert bank.similarity("arif", (1.0, 0.0)) > 0.99


def test_robust_identity_fusion_uses_positive_quality_weights():
    bank = CharacterIdentityEmbeddingBank()
    entry = bank.build_character_robust(
        "hana",
        ((1.0, 0.0), (0.8, 0.6), (0.9, 0.4)),
        reference_weights=(10.0, 1.0, 1.0),
        min_consensus_similarity=0.70,
    )

    assert entry.reference_count == 3
    assert entry.vector[0] > 0.97


def test_robust_identity_fusion_fails_closed_without_consensus():
    bank = CharacterIdentityEmbeddingBank()
    with pytest.raises(ValueError, match="identity consensus"):
        bank.build_character_robust(
            "arif",
            ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0)),
            min_consensus_similarity=0.10,
            minimum_references=2,
        )


def test_robust_identity_fusion_validates_weights_and_reference_count():
    bank = CharacterIdentityEmbeddingBank()
    with pytest.raises(ValueError, match="insufficient references"):
        bank.build_character_robust("arif", ((1.0, 0.0),))
    with pytest.raises(ValueError, match="weights must match"):
        bank.build_character_robust(
            "arif",
            ((1.0, 0.0), (0.9, 0.1)),
            reference_weights=(1.0,),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        bank.build_character_robust(
            "arif",
            ((1.0, 0.0), (0.9, 0.1)),
            reference_weights=(1.0, 0.0),
        )
