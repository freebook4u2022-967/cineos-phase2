"""Production identity coverage gate and reference collection recommendations."""

from __future__ import annotations

from dataclasses import dataclass

from .identity_coverage import IdentityCoverageAnalyzer, IdentityCoverageReport
from .training import NativeDatasetManifest


@dataclass(frozen=True, slots=True)
class IdentityGatePolicy:
    minimum_coverage_score: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_coverage_score <= 1.0:
            raise ValueError("minimum_coverage_score must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class CharacterCollectionRecommendation:
    character_id: str
    current_score: float
    required_shots: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IdentityGateDecision:
    allowed: bool
    report: IdentityCoverageReport
    recommendations: tuple[CharacterCollectionRecommendation, ...]


class IdentityTrainingGate:
    def __init__(self, policy: IdentityGatePolicy | None = None) -> None:
        self.policy = policy or IdentityGatePolicy()
        self.analyzer = IdentityCoverageAnalyzer()

    def evaluate(self, manifest: NativeDatasetManifest) -> IdentityGateDecision:
        report = self.analyzer.analyze(manifest)
        recommendations = []
        for character in report.characters:
            if character.coverage_score >= self.policy.minimum_coverage_score:
                continue
            shots = tuple(
                [f"capture {view} reference" for view in character.missing_views]
                + [f"capture {variation} variation" for variation in character.missing_variations]
            )
            recommendations.append(
                CharacterCollectionRecommendation(
                    character.character_id,
                    character.coverage_score,
                    shots,
                )
            )
        return IdentityGateDecision(not recommendations, report, tuple(recommendations))

    def require(self, manifest: NativeDatasetManifest) -> IdentityCoverageReport:
        decision = self.evaluate(manifest)
        if not decision.allowed:
            names = ", ".join(item.character_id for item in decision.recommendations)
            raise ValueError(f"identity coverage below production threshold: {names}")
        return decision.report
