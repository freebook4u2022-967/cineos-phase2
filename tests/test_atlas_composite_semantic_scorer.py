from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cineos.atlas.artifact_video_observer import RGBVideoSample
from cineos.atlas.composite_semantic_scorer import (
    COMPOSITE_SEMANTIC_SCORER_SCHEMA,
    CompositeSemanticScorerError,
    CompositeSemanticVideoScorer,
)


class _Scorer:
    semantic_measurement_evidence = True

    def __init__(self, metrics, *, name: str, measured_metrics=None) -> None:
        self.metrics = dict(metrics)
        self.name = name
        self.measured_metrics = (
            None if measured_metrics is None else list(measured_metrics)
        )
        self.calls = 0

    def runtime_provenance(self):
        provenance = {
            "schema": f"test-{self.name}/0.1",
            "origin": "external_pretrained",
            "production_measurement_evidence": True,
        }
        if self.measured_metrics is not None:
            provenance["measured_metrics"] = list(self.measured_metrics)
        return provenance

    def __call__(self, sample, *, artifact, shot, attempt_index):
        assert sample.frames
        assert artifact == Path("shot.mp4")
        assert shot.shot_id == "s01"
        assert attempt_index == 2
        self.calls += 1
        return dict(self.metrics)


def _sample() -> RGBVideoSample:
    return RGBVideoSample(width=1, height=1, frames=(b"\x00\x00\x00", b"\x01\x01\x01"))


def test_composite_merges_primary_and_specialist_metrics_with_provenance() -> None:
    primary = _Scorer(
        {"identity_similarity": 0.91, "motion_quality": 0.84}, name="siglip-motion"
    )
    specialist = _Scorer(
        {"anatomy_quality": 0.82, "physics_plausibility": 0.79},
        name="qwen",
        measured_metrics=("anatomy_quality", "physics_plausibility"),
    )
    composite = CompositeSemanticVideoScorer(primary, [specialist])

    metrics = composite(
        _sample(),
        artifact=Path("shot.mp4"),
        shot=SimpleNamespace(shot_id="s01"),
        attempt_index=2,
    )

    assert metrics == {
        "identity_similarity": 0.91,
        "motion_quality": 0.84,
        "anatomy_quality": 0.82,
        "physics_plausibility": 0.79,
    }
    assert primary.calls == 1
    assert specialist.calls == 1
    provenance = composite.runtime_provenance()
    assert provenance["schema"] == COMPOSITE_SEMANTIC_SCORER_SCHEMA
    assert provenance["origin"] == "cineos_composed_measurement_pipeline"
    assert provenance["production_measurement_evidence"] is True
    assert [item["provenance"]["origin"] for item in provenance["components"]] == [
        "external_pretrained",
        "external_pretrained",
    ]


def test_composite_rejects_specialist_replacing_primary_metric() -> None:
    primary = _Scorer(
        {"identity_similarity": 0.9, "motion_quality": 0.8}, name="primary"
    )
    specialist = _Scorer(
        {"identity_similarity": 0.2},
        name="specialist",
        measured_metrics=("identity_similarity",),
    )

    with pytest.raises(CompositeSemanticScorerError, match="core/observer metric"):
        CompositeSemanticVideoScorer(primary, [specialist])


def test_composite_rejects_duplicate_specialist_metric_ownership() -> None:
    primary = _Scorer(
        {"identity_similarity": 0.9, "motion_quality": 0.8}, name="primary"
    )
    left = _Scorer(
        {"anatomy_quality": 0.8},
        name="left",
        measured_metrics=("anatomy_quality",),
    )
    right = _Scorer(
        {"anatomy_quality": 0.9},
        name="right",
        measured_metrics=("anatomy_quality",),
    )

    with pytest.raises(
        CompositeSemanticScorerError, match="duplicates metric ownership"
    ):
        CompositeSemanticVideoScorer(primary, [left, right])


def test_composite_rejects_primary_missing_identity_or_motion() -> None:
    primary = _Scorer({"identity_similarity": 0.9}, name="primary")
    composite = CompositeSemanticVideoScorer(primary, [])

    with pytest.raises(CompositeSemanticScorerError, match="motion_quality"):
        composite(
            _sample(),
            artifact=Path("shot.mp4"),
            shot=SimpleNamespace(shot_id="s01"),
            attempt_index=2,
        )


def test_composite_rejects_unattested_or_anonymous_production_component() -> None:
    class _Anonymous:
        semantic_measurement_evidence = True

        def __call__(self, *args, **kwargs):
            return {"identity_similarity": 0.9, "motion_quality": 0.8}

    with pytest.raises(CompositeSemanticScorerError, match="runtime_provenance"):
        CompositeSemanticVideoScorer(_Anonymous(), [])

    unattested = _Scorer(
        {"identity_similarity": 0.9, "motion_quality": 0.8}, name="primary"
    )
    unattested.semantic_measurement_evidence = False
    with pytest.raises(TypeError, match="attest measured evidence"):
        CompositeSemanticVideoScorer(unattested, [])
