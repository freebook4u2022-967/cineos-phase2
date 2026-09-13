"""Regression tests for specialist semantic QC on the quality-first path."""

from pathlib import Path

import pytest

from cineos.atlas import quality_first_gpu_benchmark_cli as cli
from cineos.atlas.artifact_video_observer import ArtifactVideoMetricObserver
from cineos.atlas.composite_semantic_scorer import CompositeSemanticVideoScorer
from cineos.atlas.sequence_quality import ArtifactMeasuredSequenceQualityEvaluator


class _PrimaryScorer:
    semantic_measurement_evidence = True

    def runtime_provenance(self):
        return {
            "schema": "test-primary/0.1",
            "origin": "external_pretrained",
            "production_measurement_evidence": True,
        }

    def __call__(self, sample, *, artifact: Path, shot, attempt_index: int):
        return {"identity_similarity": 0.95, "motion_quality": 0.91}

    def _pil_frames(self, sample):
        return ["frame"]

    def _encode_images(self, frames):
        return [[0.25, 0.75]]


class _SpecialistScorer:
    semantic_measurement_evidence = True

    def runtime_provenance(self):
        return {
            "schema": "test-specialist/0.1",
            "origin": "external_pretrained",
            "production_measurement_evidence": True,
            "measured_metrics": ["anatomy_quality"],
        }

    def __call__(self, sample, *, artifact: Path, shot, attempt_index: int):
        return {"anatomy_quality": 0.89}


class _DialogueShot:
    metadata = {"benchmark_challenges": ["dialogue"]}


class _UnusedSampler:
    def __call__(self, artifact):  # pragma: no cover - construction-only fixture
        raise AssertionError("sampler should not run in wiring test")


def _base_evaluator(primary):
    observer = ArtifactVideoMetricObserver(primary, sampler=_UnusedSampler())
    return ArtifactMeasuredSequenceQualityEvaluator(observer)


def test_quality_first_composes_pinned_specialist_around_primary(monkeypatch):
    primary = _PrimaryScorer()
    specialist = _SpecialistScorer()
    base = _base_evaluator(primary)
    monkeypatch.setattr(cli, "_production_quality_evaluator", lambda *args: base)
    monkeypatch.setattr(cli, "Qwen25VLSemanticJudge", lambda: specialist)

    evaluator = cli._production_semantic_quality_evaluator([], None)

    scorer = evaluator.metric_extractor.semantic_scorer
    assert isinstance(scorer, CompositeSemanticVideoScorer)
    assert scorer.primary is primary
    assert scorer.specialists == (specialist,)
    provenance = scorer.runtime_provenance()
    assert provenance["origin"] == "cineos_composed_measurement_pipeline"
    assert provenance["components"][1]["provenance"]["origin"] == "external_pretrained"


def test_dialogue_shot_fails_before_render_without_av_sync_runtime(monkeypatch):
    base = _base_evaluator(_PrimaryScorer())
    monkeypatch.setattr(cli, "_production_quality_evaluator", lambda *args: base)

    with pytest.raises(cli.GPUProductionBenchmarkCLIError, match="LatentSync"):
        cli._production_semantic_quality_evaluator([_DialogueShot()], None)


def test_partial_av_sync_runtime_fails_closed(monkeypatch):
    base = _base_evaluator(_PrimaryScorer())
    monkeypatch.setattr(cli, "_production_quality_evaluator", lambda *args: base)

    with pytest.raises(cli.GPUProductionBenchmarkCLIError, match="LatentSync"):
        cli._production_semantic_quality_evaluator(
            [],
            None,
            latentsync_repository_root="/verified/LatentSync",
        )


def test_cli_accepts_explicit_verified_av_sync_dependency_arguments():
    args = cli._parser().parse_args(
        [
            "--requests",
            "requests.json",
            "--reference-manifest",
            "references.json",
            "--output-dir",
            "out",
            "--latentsync-repository",
            "/verified/LatentSync",
            "--latentsync-checkpoint",
            "/verified/syncnet.pt",
            "--latentsync-checkpoint-sha256",
            "a" * 64,
        ]
    )

    assert args.latentsync_repository == "/verified/LatentSync"
    assert args.latentsync_checkpoint == "/verified/syncnet.pt"
    assert args.latentsync_checkpoint_sha256 == "a" * 64


def test_transition_qc_reuses_primary_encoder_from_composite(monkeypatch):
    primary = _PrimaryScorer()
    specialist = _SpecialistScorer()
    composite = CompositeSemanticVideoScorer(primary, (specialist,))
    evaluator = ArtifactMeasuredSequenceQualityEvaluator(
        ArtifactVideoMetricObserver(composite, sampler=_UnusedSampler())
    )
    captured = {}

    class _TransitionObserver:
        production_measurement_evidence = True
        observer_id = "test-transition-observer/0.1"

        def __init__(self, adapter):
            captured["primary"] = adapter.scorer

        def __call__(self, *args, **kwargs):  # pragma: no cover - construction-only
            raise AssertionError("transition observer should not run in wiring test")

    monkeypatch.setattr(cli, "SigLIP2ArtifactTransitionObserver", _TransitionObserver)

    cli._production_transition_evaluator(evaluator)

    assert captured["primary"] is primary


def test_transition_qc_reuses_primary_encoder_from_challenge_ensemble(monkeypatch):
    primary = _PrimaryScorer()

    class _Component:
        name = "core_identity_motion"
        scorer = primary

    class _Ensemble:
        components = (_Component(),)

    evaluator = ArtifactMeasuredSequenceQualityEvaluator(
        ArtifactVideoMetricObserver(_PrimaryScorer(), sampler=_UnusedSampler())
    )
    evaluator.metric_extractor.semantic_scorer = _Ensemble()
    captured = {}

    class _TransitionObserver:
        production_measurement_evidence = True
        observer_id = "test-transition-observer/0.1"

        def __init__(self, adapter):
            captured["primary"] = adapter.scorer

        def __call__(self, *args, **kwargs):  # pragma: no cover - construction-only
            raise AssertionError("transition observer should not run in wiring test")

    monkeypatch.setattr(cli, "SigLIP2ArtifactTransitionObserver", _TransitionObserver)

    cli._production_transition_evaluator(evaluator)

    assert captured["primary"] is primary


def test_nonstandard_injected_evaluator_remains_backward_compatible(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(cli, "_production_quality_evaluator", lambda *args: sentinel)

    assert cli._production_semantic_quality_evaluator([], None) is sentinel
