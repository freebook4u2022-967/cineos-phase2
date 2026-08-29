"""Fail-closed acceptance policy for the connected CINEOS video benchmark.

The execution benchmark intentionally permits reduced/custom suites for development.
This module is the production claim boundary: it prevents a renderer from being
reported as competitively validated unless the run covers the difficult connected
shot cases, declares its model provenance, produces real artifacts, and passes
measured visual evaluation for every shot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .competitive_benchmark import CompetitiveBenchmarkReport

SEEDANCE_STYLE_REQUIRED_CHALLENGES = frozenset(
    {
        "identity_consistency",
        "multi_character_interaction",
        "hands_anatomy",
        "walking",
        "running",
        "dialogue",
        "fast_camera_movement",
        "lighting_change",
        "physics",
        "long_range_continuity",
    }
)

# A generic visual-quality score is useful but cannot substantiate a claim that the
# renderer survived specific cinematic failure modes. Each required challenge must
# therefore expose at least one explicitly named measured metric on every tagged shot.
# The names are intentionally model/evaluator neutral: an evaluator may emit
# additional metrics, but it must expose these claim dimensions before the
# competitive gate can pass.
SEEDANCE_STYLE_CHALLENGE_METRICS: dict[str, frozenset[str]] = {
    "identity_consistency": frozenset({"identity_similarity"}),
    "multi_character_interaction": frozenset({"interaction_quality"}),
    "hands_anatomy": frozenset({"anatomy_quality"}),
    "walking": frozenset({"motion_naturalness"}),
    "running": frozenset({"motion_naturalness"}),
    "dialogue": frozenset({"dialogue_lipsync"}),
    "fast_camera_movement": frozenset({"camera_geometry_stability"}),
    "lighting_change": frozenset({"lighting_consistency"}),
    "physics": frozenset({"physics_plausibility"}),
    "long_range_continuity": frozenset({"long_range_continuity"}),
}


@dataclass(frozen=True, slots=True)
class CompetitiveAcceptancePolicy:
    """Minimum evidence required before making a competitive-quality claim."""

    min_connected_shots: int = 10
    required_challenges: frozenset[str] = SEEDANCE_STYLE_REQUIRED_CHALLENGES
    require_declared_provenance: bool = True
    require_model_revision: bool = True
    require_license_id: bool = True
    require_nonempty_metrics_per_shot: bool = True
    require_challenge_metric_evidence: bool = True
    min_challenge_metric_score: float = 0.80

    def __post_init__(self) -> None:
        if self.min_connected_shots < 1:
            raise ValueError("min_connected_shots must be >= 1")
        if not self.required_challenges:
            raise ValueError("required_challenges must not be empty")
        if not 0.0 <= self.min_challenge_metric_score <= 1.0:
            raise ValueError("min_challenge_metric_score must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class CompetitiveAcceptance:
    """Auditable competitive acceptance verdict with explicit failure reasons."""

    passed: bool
    reasons: tuple[str, ...]
    covered_challenges: frozenset[str]
    missing_challenges: frozenset[str]
    evaluated_metric_names: frozenset[str]
    missing_metric_evidence: frozenset[str]
    below_threshold_challenges: frozenset[str]
    minimum_observed_challenge_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cineos-competitive-acceptance/0.3",
            "passed": self.passed,
            "reasons": list(self.reasons),
            "covered_challenges": sorted(self.covered_challenges),
            "missing_challenges": sorted(self.missing_challenges),
            "evaluated_metric_names": sorted(self.evaluated_metric_names),
            "missing_metric_evidence": sorted(self.missing_metric_evidence),
            "below_threshold_challenges": sorted(self.below_threshold_challenges),
            "minimum_observed_challenge_score": self.minimum_observed_challenge_score,
        }


def _challenge_metric_scores(
    report: CompetitiveBenchmarkReport,
    required_challenges: frozenset[str],
) -> tuple[dict[str, tuple[float, ...]], frozenset[str]]:
    """Collect required metric scores and identify incomplete tagged-shot evidence.

    A challenge is considered to have complete evidence only when every shot carrying
    that challenge tag exposes at least one of the challenge's canonical metrics.
    This prevents a single good shot from masking unmeasured failures elsewhere in a
    connected sequence.
    """

    scores: dict[str, tuple[float, ...]] = {}
    missing: set[str] = set()
    for challenge in required_challenges:
        expected = SEEDANCE_STYLE_CHALLENGE_METRICS.get(challenge)
        if not expected:
            continue
        tagged_shots = [
            shot for shot in report.shots if challenge in shot.challenge_tags
        ]
        if not tagged_shots:
            continue

        challenge_scores: list[float] = []
        complete = True
        for shot in tagged_shots:
            values = [
                float(shot.quality_metrics[name])
                for name in expected
                if name in shot.quality_metrics
            ]
            if not values:
                complete = False
                continue
            challenge_scores.append(min(values))

        if not complete:
            missing.add(challenge)
        if challenge_scores:
            scores[challenge] = tuple(challenge_scores)

    return scores, frozenset(missing)


def evaluate_competitive_acceptance(
    report: CompetitiveBenchmarkReport,
    *,
    policy: CompetitiveAcceptancePolicy | None = None,
) -> CompetitiveAcceptance:
    """Evaluate whether a report supports a competitive complete-scene claim.

    This gate deliberately does not infer quality from artifact existence. A passing
    verdict requires the benchmark's measured quality verdict as well as coverage,
    provenance and challenge-specific metric evidence. Reduced suites remain useful
    for development but cannot accidentally graduate into a Seedance-class claim.
    """

    active = policy or CompetitiveAcceptancePolicy()
    reasons: list[str] = []

    covered = frozenset(tag for shot in report.shots for tag in shot.challenge_tags)
    missing = frozenset(active.required_challenges - covered)
    metric_names = frozenset(
        name for shot in report.shots for name in shot.quality_metrics.keys()
    )

    if len(report.shots) < active.min_connected_shots:
        reasons.append(
            f"requires at least {active.min_connected_shots} connected shots; "
            f"got {len(report.shots)}"
        )
    if missing:
        reasons.append(
            "missing required challenge coverage: " + ", ".join(sorted(missing))
        )
    if not report.execution_passed:
        reasons.append("one or more benchmark shots lack real execution evidence")
    if not report.quality_validated:
        reasons.append("visual quality was not measured for every benchmark shot")
    elif not report.quality_passed:
        reasons.append("one or more benchmark shots failed measured visual quality")

    foundation = dict(report.foundation)
    if active.require_declared_provenance and not foundation.get(
        "provenance_declared", False
    ):
        reasons.append("renderer foundation provenance is not declared")
    model_id = str(foundation.get("model_id", "")).strip()
    if not model_id or model_id.lower() == "unknown":
        reasons.append("renderer foundation model_id is missing or unknown")
    if (
        active.require_model_revision
        and not str(foundation.get("revision", "")).strip()
    ):
        reasons.append("renderer foundation revision is not declared")
    if active.require_license_id and not str(foundation.get("license_id", "")).strip():
        reasons.append("renderer foundation license_id is not declared")

    if active.require_nonempty_metrics_per_shot:
        missing_metric_shots = [
            shot.shot_id
            for shot in report.shots
            if shot.quality_evaluated and not shot.quality_metrics
        ]
        if missing_metric_shots:
            reasons.append(
                "quality evaluator returned no numeric metrics for: "
                + ", ".join(missing_metric_shots)
            )

    challenge_scores, missing_metric_evidence = _challenge_metric_scores(
        report, active.required_challenges
    )
    if not active.require_challenge_metric_evidence:
        missing_metric_evidence = frozenset()
    elif missing_metric_evidence:
        reasons.append(
            "missing challenge-specific metric evidence on one or more tagged shots: "
            + ", ".join(sorted(missing_metric_evidence))
        )

    below_threshold = frozenset(
        challenge
        for challenge, values in challenge_scores.items()
        if values and min(values) < active.min_challenge_metric_score
    )
    if below_threshold:
        reasons.append(
            "challenge-specific metrics below competitive threshold "
            f"{active.min_challenge_metric_score:.2f}: "
            + ", ".join(sorted(below_threshold))
        )

    observed_scores = [
        score for values in challenge_scores.values() for score in values
    ]
    minimum_observed_score = min(observed_scores) if observed_scores else None

    return CompetitiveAcceptance(
        passed=not reasons,
        reasons=tuple(reasons),
        covered_challenges=covered,
        missing_challenges=missing,
        evaluated_metric_names=metric_names,
        missing_metric_evidence=missing_metric_evidence,
        below_threshold_challenges=below_threshold,
        minimum_observed_challenge_score=minimum_observed_score,
    )


__all__ = [
    "CompetitiveAcceptance",
    "CompetitiveAcceptancePolicy",
    "SEEDANCE_STYLE_CHALLENGE_METRICS",
    "SEEDANCE_STYLE_REQUIRED_CHALLENGES",
    "evaluate_competitive_acceptance",
]
