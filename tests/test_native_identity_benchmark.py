import pytest

from cineos.native_image.identity_bank import CharacterIdentityEmbeddingBank
from cineos.native_image.identity_benchmark import CrossShotIdentityBenchmark


def _bank():
    bank = CharacterIdentityEmbeddingBank()
    bank.build_character("arif", [(1.0, 0.0)])
    bank.build_character("hana", [(0.0, 1.0)])
    return bank


def test_identity_benchmark_rewards_correct_character_embeddings():
    report = CrossShotIdentityBenchmark(_bank()).evaluate(
        [("arif", (1.0, 0.0)), ("hana", (0.0, 1.0))]
    )
    assert report.mean_anchor_similarity == pytest.approx(1.0)
    assert report.mean_identity_margin == pytest.approx(1.0)
    assert report.identity_consistency_score == pytest.approx(1.0)


def test_identity_benchmark_penalizes_character_confusion():
    report = CrossShotIdentityBenchmark(_bank()).evaluate([("arif", (0.0, 1.0))])
    assert report.shots[0].anchor_similarity == pytest.approx(0.0)
    assert report.shots[0].identity_margin == pytest.approx(-1.0)
    assert report.identity_consistency_score == pytest.approx(0.0)


def test_identity_benchmark_rejects_unknown_character():
    with pytest.raises(ValueError, match="azman"):
        CrossShotIdentityBenchmark(_bank()).evaluate([("azman", (1.0, 0.0))])
