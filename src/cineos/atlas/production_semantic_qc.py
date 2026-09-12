"""Production semantic-QC capability contract for competitive CINEOS runs.

This module bridges the benchmark challenge contract to the semantic scorer ensemble.
It deliberately does not invent difficult-case scores. A production competitive run
must have an explicitly owned, production-attested semantic measurement for every
challenge-specific metric it declares before expensive video generation should begin.

The default core scorer may provide identity and motion measurements. Specialist
anatomy, locomotion, interaction, camera, lighting, physics, and lip-sync measurements
remain separate components with their own provenance. External pretrained foundations
are valid components when their provenance is transparent; they are never relabelled
as CINEOS-native models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .semantic_video_ensemble import (
    ProductionSemanticScorerEnsemble,
    SemanticScorerComponent,
    SemanticScorerEnsembleError,
)
from .sequence_quality import CHALLENGE_METRIC_REQUIREMENTS

PRODUCTION_SEMANTIC_QC_SCHEMA = "cineos-production-semantic-qc-capabilities/0.1"
CORE_SEMANTIC_METRICS = ("identity_similarity", "motion_quality")
_CHALLENGE_METADATA_KEYS = ("competitive_challenges", "benchmark_challenges")


class ProductionSemanticQCError(RuntimeError):
    """Raised when competitive semantic QC cannot be proven before production."""


def required_semantic_metrics(shots: Sequence[Any]) -> frozenset[str]:
    """Return semantic metrics required by the difficult cases declared by ``shots``.

    Unknown challenge names are intentionally ignored here because benchmark request
    validation owns the challenge vocabulary. This helper only translates known
    quality-contract declarations into scorer capabilities without duplicating request
    validation policy.
    """

    required = set(CORE_SEMANTIC_METRICS)
    for shot in shots:
        metadata = getattr(shot, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        for key in _CHALLENGE_METADATA_KEYS:
            raw = metadata.get(key)
            if raw is None or isinstance(raw, (str, bytes)):
                continue
            try:
                values = tuple(raw)
            except TypeError:
                continue
            for challenge in values:
                if not isinstance(challenge, str):
                    continue
                metric = CHALLENGE_METRIC_REQUIREMENTS.get(challenge.strip())
                if metric is not None:
                    required.add(metric)
    return frozenset(required)


def validate_production_semantic_capabilities(
    scorer: ProductionSemanticScorerEnsemble,
    shots: Sequence[Any],
) -> frozenset[str]:
    """Fail closed unless every required metric has an attested scorer owner.

    The check happens against declared ownership before artifact decoding or model
    rendering. This prevents a 5-10 shot CUDA benchmark from spending substantial GPU
    time only to discover that anatomy, lip-sync, physics, or another required metric
    could never have been measured by the configured QC stack.
    """

    if not isinstance(scorer, ProductionSemanticScorerEnsemble):
        raise TypeError("scorer must be ProductionSemanticScorerEnsemble")
    if scorer.semantic_measurement_evidence is not True:
        raise ProductionSemanticQCError(
            "production semantic QC requires every scorer component to attest real "
            "semantic measurement evidence"
        )

    required = required_semantic_metrics(shots)
    available = frozenset(scorer.metric_owners)
    missing = sorted(required - available)
    if missing:
        raise ProductionSemanticQCError(
            "production semantic QC is missing real measured scorer capability for: "
            + ", ".join(missing)
        )
    return required


def build_production_semantic_scorer(
    core_scorer: Any,
    shots: Sequence[Any],
    *,
    specialists: Sequence[SemanticScorerComponent] = (),
) -> ProductionSemanticScorerEnsemble:
    """Compose core plus specialist scorers and validate production metric coverage.

    ``core_scorer`` is restricted to identity and motion ownership. Difficult-case
    metrics must come from separately named specialist components, making provenance
    substitution visible and preventing a generic visual encoder from being relabelled
    as anatomy, physics, interaction, or lip-sync evidence.
    """

    if not callable(core_scorer):
        raise TypeError("core_scorer must be callable")
    if isinstance(specialists, (str, bytes)) or not isinstance(specialists, Sequence):
        raise TypeError("specialists must be a sequence of SemanticScorerComponent")
    materialized = tuple(specialists)
    if any(not isinstance(item, SemanticScorerComponent) for item in materialized):
        raise TypeError("specialists must contain only SemanticScorerComponent values")

    core = SemanticScorerComponent(
        name="core_identity_motion",
        scorer=core_scorer,
        measured_metrics=CORE_SEMANTIC_METRICS,
    )
    try:
        ensemble = ProductionSemanticScorerEnsemble((core, *materialized))
    except (TypeError, ValueError, SemanticScorerEnsembleError) as exc:
        raise ProductionSemanticQCError(
            f"cannot compose production semantic QC scorers: {exc}"
        ) from exc
    validate_production_semantic_capabilities(ensemble, shots)
    return ensemble


__all__ = [
    "CORE_SEMANTIC_METRICS",
    "PRODUCTION_SEMANTIC_QC_SCHEMA",
    "ProductionSemanticQCError",
    "build_production_semantic_scorer",
    "required_semantic_metrics",
    "validate_production_semantic_capabilities",
]
