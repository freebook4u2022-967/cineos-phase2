"""Cross-shot identity benchmark for CINEOS character continuity."""

from __future__ import annotations

from dataclasses import dataclass

from .identity_bank import CharacterIdentityEmbeddingBank


@dataclass(frozen=True, slots=True)
class IdentityShotScore:
    character_id: str
    anchor_similarity: float
    strongest_impostor_similarity: float
    identity_margin: float


@dataclass(frozen=True, slots=True)
class IdentityBenchmarkReport:
    shots: tuple[IdentityShotScore, ...]

    @property
    def mean_anchor_similarity(self) -> float:
        if not self.shots:
            return 0.0
        return sum(item.anchor_similarity for item in self.shots) / len(self.shots)

    @property
    def mean_identity_margin(self) -> float:
        if not self.shots:
            return 0.0
        return sum(item.identity_margin for item in self.shots) / len(self.shots)

    @property
    def identity_consistency_score(self) -> float:
        similarity = max(0.0, min(1.0, self.mean_anchor_similarity))
        margin = max(0.0, min(1.0, (self.mean_identity_margin + 1.0) / 2.0))
        return 0.75 * similarity + 0.25 * margin


class CrossShotIdentityBenchmark:
    """Score generated shot embeddings against character anchors and impostors."""

    def __init__(self, bank: CharacterIdentityEmbeddingBank) -> None:
        self.bank = bank

    def evaluate(self, shots) -> IdentityBenchmarkReport:
        scores = []
        characters = self.bank.characters()
        for character_id, embedding in shots:
            if character_id not in characters:
                raise ValueError(f"identity anchor missing for: {character_id}")
            positive = self.bank.similarity(character_id, embedding)
            impostors = [
                self.bank.similarity(other, embedding)
                for other in characters
                if other != character_id
            ]
            strongest_impostor = max(impostors) if impostors else -1.0
            scores.append(
                IdentityShotScore(
                    character_id=character_id,
                    anchor_similarity=positive,
                    strongest_impostor_similarity=strongest_impostor,
                    identity_margin=positive - strongest_impostor,
                )
            )
        return IdentityBenchmarkReport(tuple(scores))
