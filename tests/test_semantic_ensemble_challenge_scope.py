from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cineos.atlas.artifact_video_observer import RGBVideoSample
from cineos.atlas.semantic_video_ensemble import (
    ProductionSemanticScorerEnsemble,
    SemanticScorerComponent,
    SemanticScorerEnsembleError,
)


class _CountingScorer:
    semantic_measurement_evidence = True

    def __init__(self, metrics: dict[str, float]) -> None:
        self.metrics = dict(metrics)
        self.calls = 0

    def __call__(self, sample, *, artifact, shot, attempt_index):
        del sample, artifact, shot, attempt_index
        self.calls += 1
        return dict(self.metrics)

    def runtime_provenance(self):
        return {
            "schema": "test-counting-scorer/0.1",
            "origin": "external_pretrained",
            "production_measurement_evidence": True,
        }


def _sample() -> RGBVideoSample:
    return RGBVideoSample(
        width=1,
        height=1,
        frames=(b"\x00\x00\x00", b"\x01\x01\x01"),
    )


def test_dialogue_specialist_is_not_called_for_non_dialogue_shot() -> None:
    core = _CountingScorer({"identity_similarity": 0.9, "motion_quality": 0.8})
    av = _CountingScorer({"dialogue_lip_sync": 1.0})
    ensemble = ProductionSemanticScorerEnsemble(
        (
            SemanticScorerComponent(
                "core",
                core,
                ("identity_similarity", "motion_quality"),
            ),
            SemanticScorerComponent(
                "av",
                av,
                ("dialogue_lip_sync",),
                required_challenges=("dialogue", "dialogue_lip_sync"),
            ),
        )
    )

    metrics = ensemble(
        _sample(),
        artifact=Path("shot.mp4"),
        shot=SimpleNamespace(metadata={"benchmark_challenges": ["hands_anatomy"]}),
        attempt_index=0,
    )

    assert metrics == {"identity_similarity": 0.9, "motion_quality": 0.8}
    assert core.calls == 1
    assert av.calls == 0


@pytest.mark.parametrize("challenge", ["dialogue", "dialogue_lip_sync"])
def test_dialogue_specialist_runs_for_supported_dialogue_aliases(challenge: str) -> None:
    av = _CountingScorer({"dialogue_lip_sync": 1.0})
    ensemble = ProductionSemanticScorerEnsemble(
        (
            SemanticScorerComponent(
                "av",
                av,
                ("dialogue_lip_sync",),
                required_challenges=("dialogue", "dialogue_lip_sync"),
            ),
        )
    )

    metrics = ensemble(
        _sample(),
        artifact=Path("shot.mp4"),
        shot=SimpleNamespace(metadata={"competitive_challenges": [challenge]}),
        attempt_index=0,
    )

    assert metrics == {"dialogue_lip_sync": 1.0}
    assert av.calls == 1
    component = ensemble.runtime_provenance()["components"][0]
    assert component["required_challenges"] == ["dialogue", "dialogue_lip_sync"]


def test_malformed_challenge_metadata_fails_closed_before_specialist_execution() -> None:
    av = _CountingScorer({"dialogue_lip_sync": 1.0})
    ensemble = ProductionSemanticScorerEnsemble(
        (
            SemanticScorerComponent(
                "av",
                av,
                ("dialogue_lip_sync",),
                required_challenges=("dialogue",),
            ),
        )
    )

    with pytest.raises(SemanticScorerEnsembleError, match="must be a sequence"):
        ensemble(
            _sample(),
            artifact=Path("shot.mp4"),
            shot=SimpleNamespace(metadata={"benchmark_challenges": "dialogue"}),
            attempt_index=0,
        )
    assert av.calls == 0
