from pathlib import Path

import pytest

from cineos.atlas.composite_semantic_scorer import (
    CompositeSemanticScorerError,
    CompositeSemanticVideoScorer,
)


class _Primary:
    semantic_measurement_evidence = True

    def runtime_provenance(self):
        return {
            "schema": "test-primary/0.1",
            "origin": "external_pretrained",
            "production_measurement_evidence": True,
        }

    def __call__(self, sample, *, artifact, shot, attempt_index):
        return {"identity_similarity": 0.9, "motion_quality": 0.8}


class _Specialist:
    semantic_measurement_evidence = True

    def __init__(self, declared, emitted):
        self.declared = list(declared)
        self.emitted = dict(emitted)

    def runtime_provenance(self):
        return {
            "schema": "test-specialist/0.1",
            "origin": "external_pretrained",
            "production_measurement_evidence": True,
            "measured_metrics": list(self.declared),
        }

    def __call__(self, sample, *, artifact, shot, attempt_index):
        return dict(self.emitted)


def _score(composite):
    return composite(None, artifact=Path("shot.mp4"), shot=object(), attempt_index=0)


def test_specialist_metric_must_be_attested_by_component_provenance():
    composite = CompositeSemanticVideoScorer(
        _Primary(),
        [_Specialist(["anatomy_quality"], {"object_interaction_quality": 0.7})],
    )

    with pytest.raises(CompositeSemanticScorerError, match="not attested by provenance"):
        _score(composite)


def test_specialist_metric_ownership_must_be_unique_before_inference():
    with pytest.raises(CompositeSemanticScorerError, match="duplicates metric ownership"):
        CompositeSemanticVideoScorer(
            _Primary(),
            [
                _Specialist(["anatomy_quality"], {"anatomy_quality": 0.8}),
                _Specialist(["anatomy_quality"], {"anatomy_quality": 0.9}),
            ],
        )


def test_specialist_must_declare_nonempty_metric_ownership():
    with pytest.raises(CompositeSemanticScorerError, match="non-empty measured_metrics"):
        CompositeSemanticVideoScorer(_Primary(), [_Specialist([], {})])


def test_specialist_provenance_cannot_claim_core_metric():
    with pytest.raises(CompositeSemanticScorerError, match="core/observer metric"):
        CompositeSemanticVideoScorer(
            _Primary(),
            [_Specialist(["identity_similarity"], {"identity_similarity": 0.4})],
        )


def test_valid_specialist_metrics_preserve_exact_ownership_provenance():
    composite = CompositeSemanticVideoScorer(
        _Primary(),
        [
            _Specialist(
                ["anatomy_quality", "object_interaction_quality"],
                {"anatomy_quality": 0.86, "object_interaction_quality": 0.81},
            )
        ],
    )

    metrics = _score(composite)
    assert metrics["anatomy_quality"] == pytest.approx(0.86)
    assert metrics["object_interaction_quality"] == pytest.approx(0.81)
    provenance = composite.runtime_provenance()
    assert provenance["ownership"]["specialists"] == [
        {
            "role": "specialist[0]",
            "measured_metrics": ["anatomy_quality", "object_interaction_quality"],
        }
    ]
