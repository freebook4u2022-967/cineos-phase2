"""Regression coverage for measured cross-shot transition QC."""

from types import SimpleNamespace

import pytest

from cineos.atlas import artifact_transition_observer as transition_observer
from cineos.atlas import quality_first_gpu_benchmark_cli as quality_first
from cineos.atlas.artifact_video_observer import RGBVideoSample
from cineos.atlas.gpu_benchmark_cli import GPUProductionBenchmarkCLIError
from cineos.atlas.sequence_quality import ArtifactMeasuredSequenceQualityEvaluator
from cineos.atlas.transition_quality import ArtifactMeasuredTransitionQualityEvaluator


class _PinnedFeatureScorer:
    semantic_measurement_evidence = True

    def _pil_frames(self, sample):
        return sample.frames

    def _encode_images(self, images):
        del images
        return ((1.0, 0.0), (0.98, 0.02))


class _ShotObserver:
    production_measurement_evidence = True
    observer_id = "test-shot-observer"

    def __init__(self, scorer):
        self.semantic_scorer = scorer

    def __call__(self, *args, **kwargs):
        raise AssertionError(
            "shot observer should not execute in this construction test"
        )


def test_quality_first_transition_gate_reuses_attested_pinned_qc_encoder():
    scorer = _PinnedFeatureScorer()
    shot_quality = ArtifactMeasuredSequenceQualityEvaluator(_ShotObserver(scorer))

    transition_quality = quality_first._production_transition_evaluator(shot_quality)

    assert isinstance(transition_quality, ArtifactMeasuredTransitionQualityEvaluator)
    assert transition_quality.production_measurement_evidence is True
    assert transition_quality.observer.production_measurement_evidence is True
    assert transition_quality.observer.feature_scorer.scorer is scorer


def test_quality_first_transition_gate_rejects_missing_semantic_scorer():
    evaluator = SimpleNamespace(metric_extractor=SimpleNamespace())

    with pytest.raises(GPUProductionBenchmarkCLIError, match="pinned semantic scorer"):
        quality_first._production_transition_evaluator(evaluator)


def test_boundary_metrics_accept_identical_visual_and_motion_handoff():
    previous = ((1.0, 0.0), (0.98, 0.02))
    current = ((0.98, 0.02), (0.96, 0.04))

    visual, motion = transition_observer._boundary_metrics(previous, current)

    assert visual == pytest.approx(1.0)
    assert motion > 0.99


def test_boundary_metrics_detect_large_cross_boundary_semantic_jump():
    previous = ((1.0, 0.0), (1.0, 0.0))
    current = ((0.0, 1.0), (0.0, 1.0))

    visual, motion = transition_observer._boundary_metrics(previous, current)

    assert visual == pytest.approx(0.5)
    assert motion == pytest.approx(0.0)


def test_injected_boundary_sampler_cannot_attest_production_measurement():
    sample = RGBVideoSample(
        width=1,
        height=1,
        frames=(b"\x00\x00\x00", b"\x01\x01\x01"),
    )

    class _InjectedSampler:
        production_measurement_evidence = True

        def __call__(self, artifact, *, tail):
            del artifact, tail
            return sample

    observer = transition_observer.SigLIP2ArtifactTransitionObserver(
        SimpleNamespace(
            semantic_measurement_evidence=True,
            encode_sample_features=lambda sample: ((1.0, 0.0), (0.99, 0.01)),
        ),
        sampler=_InjectedSampler(),
    )

    assert observer.production_measurement_evidence is False
    with pytest.raises(TypeError, match="attest measurement evidence"):
        ArtifactMeasuredTransitionQualityEvaluator(observer)
