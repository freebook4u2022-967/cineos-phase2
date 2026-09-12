from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cineos.atlas.artifact_video_observer import (
    ArtifactVideoMetricObserver,
    RGBVideoSample,
)
from cineos.atlas.semantic_video_ensemble import (
    ProductionSemanticScorerEnsemble,
    SemanticScorerComponent,
    SemanticScorerEnsembleError,
)


class _MeasuredScorer:
    semantic_measurement_evidence = True

    def __init__(self, metrics: dict[str, float], *, origin: str) -> None:
        self.metrics = dict(metrics)
        self.origin = origin

    def __call__(self, sample, *, artifact, shot, attempt_index):
        del sample, artifact, shot, attempt_index
        return dict(self.metrics)

    def runtime_provenance(self):
        return {
            "schema": f"test-{self.origin}/0.1",
            "origin": self.origin,
            "production_measurement_evidence": True,
        }


class _ResearchScorer(_MeasuredScorer):
    semantic_measurement_evidence = False

    def runtime_provenance(self):
        evidence = super().runtime_provenance()
        evidence["production_measurement_evidence"] = False
        return evidence


class _StaticSampler:
    def __call__(self, artifact: Path) -> RGBVideoSample:
        del artifact
        frame_a = bytes([32, 64, 96] * 4)
        frame_b = bytes([34, 66, 98] * 4)
        return RGBVideoSample(width=2, height=2, frames=(frame_a, frame_b))


def _ensemble() -> ProductionSemanticScorerEnsemble:
    return ProductionSemanticScorerEnsemble(
        (
            SemanticScorerComponent(
                "identity-motion",
                _MeasuredScorer(
                    {"identity_similarity": 0.91, "motion_quality": 0.88},
                    origin="external-pretrained-foundation",
                ),
                ("identity_similarity", "motion_quality"),
            ),
            SemanticScorerComponent(
                "anatomy",
                _MeasuredScorer(
                    {"anatomy_quality": 0.86},
                    origin="external-specialist-anatomy-model",
                ),
                ("anatomy_quality",),
            ),
            SemanticScorerComponent(
                "lip-sync",
                _MeasuredScorer(
                    {"dialogue_lip_sync": 0.84},
                    origin="external-specialist-lipsync-model",
                ),
                ("dialogue_lip_sync",),
            ),
        )
    )


def test_ensemble_composes_disjoint_specialist_measurements() -> None:
    ensemble = _ensemble()
    sample = _StaticSampler()(Path("ignored.mp4"))

    metrics = ensemble(
        sample,
        artifact=Path("shot.mp4"),
        shot=SimpleNamespace(),
        attempt_index=0,
    )

    assert metrics == {
        "identity_similarity": 0.91,
        "motion_quality": 0.88,
        "anatomy_quality": 0.86,
        "dialogue_lip_sync": 0.84,
    }
    assert ensemble.semantic_measurement_evidence is True


def test_ensemble_provenance_preserves_metric_ownership_and_external_origin() -> None:
    provenance = _ensemble().runtime_provenance()

    assert provenance["origin"] == "cineos-composition-of-declared-semantic-scorers"
    assert provenance["metric_owners"] == {
        "identity_similarity": "identity-motion",
        "motion_quality": "identity-motion",
        "anatomy_quality": "anatomy",
        "dialogue_lip_sync": "lip-sync",
    }
    assert provenance["components"][1]["scorer"]["origin"] == (
        "external-specialist-anatomy-model"
    )
    assert provenance["components"][2]["measured_metrics"] == ["dialogue_lip_sync"]


def test_ensemble_rejects_duplicate_metric_ownership() -> None:
    scorer = _MeasuredScorer({"motion_quality": 0.9}, origin="external-a")
    with pytest.raises(ValueError, match="owned by both"):
        ProductionSemanticScorerEnsemble(
            (
                SemanticScorerComponent("a", scorer, ("motion_quality",)),
                SemanticScorerComponent("b", scorer, ("motion_quality",)),
            )
        )


@pytest.mark.parametrize(
    ("metrics", "declared", "match"),
    [
        (
            {"identity_similarity": 0.9},
            ("identity_similarity", "motion_quality"),
            "missing",
        ),
        (
            {"identity_similarity": 0.9, "motion_quality": 0.8},
            ("identity_similarity",),
            "undeclared",
        ),
    ],
)
def test_ensemble_rejects_component_output_that_violates_metric_ownership(
    metrics: dict[str, float], declared: tuple[str, ...], match: str
) -> None:
    ensemble = ProductionSemanticScorerEnsemble(
        (
            SemanticScorerComponent(
                "component",
                _MeasuredScorer(metrics, origin="external-test"),
                declared,
            ),
        )
    )

    with pytest.raises(SemanticScorerEnsembleError, match=match):
        ensemble(
            _StaticSampler()(Path("ignored.mp4")),
            artifact=Path("shot.mp4"),
            shot=SimpleNamespace(),
            attempt_index=0,
        )


def test_research_component_prevents_production_attestation() -> None:
    ensemble = ProductionSemanticScorerEnsemble(
        (
            SemanticScorerComponent(
                "research-anatomy",
                _ResearchScorer(
                    {"anatomy_quality": 0.9},
                    origin="research-only-model",
                ),
                ("anatomy_quality",),
            ),
        )
    )

    assert ensemble.semantic_measurement_evidence is False
    assert ensemble.runtime_provenance()["production_measurement_evidence"] is False


def test_ensemble_requires_runtime_provenance_for_auditable_components() -> None:
    class NoProvenance:
        semantic_measurement_evidence = True

        def __call__(self, sample, *, artifact, shot, attempt_index):
            del sample, artifact, shot, attempt_index
            return {"anatomy_quality": 0.9}

    ensemble = ProductionSemanticScorerEnsemble(
        (
            SemanticScorerComponent(
                "anonymous-anatomy",
                NoProvenance(),
                ("anatomy_quality",),
            ),
        )
    )

    with pytest.raises(SemanticScorerEnsembleError, match="runtime_provenance"):
        ensemble.runtime_provenance()


def test_artifact_observer_serializes_ensemble_provenance_and_specialist_metrics(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "shot.mp4"
    artifact.write_bytes(b"real-render-placeholder-for-observer-binding")
    observer = ArtifactVideoMetricObserver(_ensemble(), sampler=_StaticSampler())

    measurement = observer(
        str(artifact),
        shot=SimpleNamespace(),
        attempt_index=1,
    )

    assert measurement["production_measurement_evidence"] is True
    assert measurement["metrics"]["anatomy_quality"] == 0.86
    assert measurement["metrics"]["dialogue_lip_sync"] == 0.84
    provenance = measurement["semantic_scorer"]
    assert provenance["metric_owners"]["anatomy_quality"] == "anatomy"
    assert provenance["components"][0]["scorer"]["origin"] == (
        "external-pretrained-foundation"
    )
