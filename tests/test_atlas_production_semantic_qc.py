from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from cineos.atlas.production_semantic_qc import (
    CORE_SEMANTIC_METRICS,
    ProductionSemanticQCError,
    build_production_semantic_scorer,
    required_semantic_metrics,
)
from cineos.atlas.semantic_video_ensemble import SemanticScorerComponent
from cineos.atlas.sequence_quality import CHALLENGE_METRIC_REQUIREMENTS


class _MeasuredScorer:
    def __init__(self, metrics: tuple[str, ...], *, attested: bool = True) -> None:
        self.metrics = metrics
        self.semantic_measurement_evidence = attested

    def runtime_provenance(self) -> dict[str, Any]:
        return {
            "schema": "test-measured-semantic-scorer/0.1",
            "origin": "test-double-not-production-model",
            "production_measurement_evidence": self.semantic_measurement_evidence,
        }

    def __call__(self, sample: Any, **kwargs: Any) -> dict[str, float]:
        del sample, kwargs
        return {metric: 0.9 for metric in self.metrics}


def _competitive_shot() -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "competitive_challenges": [
                "multi_character_interaction",
                "hands_anatomy",
                "walking_running",
                "object_interaction",
                "fast_camera_movement",
                "lighting_changes",
                "physics",
                "dialogue_lip_sync",
            ]
        }
    )


def test_required_semantic_metrics_match_quality_contract() -> None:
    expected = set(CORE_SEMANTIC_METRICS)
    expected.update(CHALLENGE_METRIC_REQUIREMENTS.values())

    assert required_semantic_metrics((_competitive_shot(),)) == frozenset(expected)


def test_production_semantic_qc_fails_before_render_when_specialists_missing() -> None:
    core = _MeasuredScorer(CORE_SEMANTIC_METRICS)

    with pytest.raises(ProductionSemanticQCError) as exc_info:
        build_production_semantic_scorer(core, (_competitive_shot(),))

    message = str(exc_info.value)
    assert "anatomy_quality" in message
    assert "dialogue_lip_sync" in message
    assert "physics_plausibility" in message


def test_production_semantic_qc_accepts_complete_attested_specialist_set() -> None:
    core = _MeasuredScorer(CORE_SEMANTIC_METRICS)
    specialist_metrics = tuple(dict.fromkeys(CHALLENGE_METRIC_REQUIREMENTS.values()))
    specialist = SemanticScorerComponent(
        name="test_specialists",
        scorer=_MeasuredScorer(specialist_metrics),
        measured_metrics=specialist_metrics,
    )

    scorer = build_production_semantic_scorer(
        core,
        (_competitive_shot(),),
        specialists=(specialist,),
    )

    assert scorer.semantic_measurement_evidence is True
    assert set(scorer.metric_owners) == set(CORE_SEMANTIC_METRICS) | set(
        specialist_metrics
    )
    provenance = scorer.runtime_provenance()
    assert provenance["production_measurement_evidence"] is True
    assert provenance["components"][1]["scorer"]["origin"] == (
        "test-double-not-production-model"
    )


def test_production_semantic_qc_rejects_unattested_specialist() -> None:
    core = _MeasuredScorer(CORE_SEMANTIC_METRICS)
    specialist = SemanticScorerComponent(
        name="unattested_anatomy",
        scorer=_MeasuredScorer(("anatomy_quality",), attested=False),
        measured_metrics=("anatomy_quality",),
    )
    shot = SimpleNamespace(metadata={"competitive_challenges": ["hands_anatomy"]})

    with pytest.raises(ProductionSemanticQCError, match="attest real semantic"):
        build_production_semantic_scorer(core, (shot,), specialists=(specialist,))


def test_benchmark_dialogue_alias_requires_same_lip_sync_metric() -> None:
    shot = SimpleNamespace(metadata={"benchmark_challenges": ["dialogue"]})

    assert required_semantic_metrics((shot,)) == frozenset(
        {"identity_similarity", "motion_quality", "dialogue_lip_sync"}
    )
