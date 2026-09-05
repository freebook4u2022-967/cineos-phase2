import hashlib

import pytest

from cineos.atlas.artifact_video_observer import (
    ArtifactVideoMetricObserver,
    RGBVideoSample,
    VideoArtifactObservationError,
)
from cineos.atlas.sequence_quality import (
    ArtifactMeasuredSequenceQualityEvaluator,
)


class Shot:
    shot_id = "shot-01"


class AttestedSemanticScorer:
    semantic_measurement_evidence = True

    def runtime_provenance(self):
        return {
            "schema": "test-semantic-scorer/0.1",
            "origin": "external-pretrained-foundation",
        }

    def __call__(self, sample, **_kwargs):
        assert len(sample.frames) == 3
        return {
            "identity_similarity": 0.94,
            "motion_quality": 0.88,
            "anatomy_quality": 0.86,
        }


def _frame(value: int, *, width: int = 2, height: int = 2) -> bytes:
    return bytes([value, value, value] * width * height)


def _sampler(_artifact):
    return RGBVideoSample(
        width=2,
        height=2,
        frames=(_frame(40), _frame(50), _frame(65)),
    )


def _semantic_scorer(sample, **_kwargs):
    assert len(sample.frames) == 3
    return {
        "identity_similarity": 0.94,
        "motion_quality": 0.88,
        "anatomy_quality": 0.86,
    }


def test_video_observer_binds_decoded_measurements_to_exact_artifact(tmp_path):
    artifact = tmp_path / "candidate.mp4"
    artifact.write_bytes(b"actual-rendered-video-container")
    observer = ArtifactVideoMetricObserver(
        AttestedSemanticScorer(),
        sampler=_sampler,
        observer_id="test-video-observer/0.1",
    )

    measurement = observer(str(artifact), shot=Shot(), attempt_index=0)

    assert (
        measurement["artifact_sha256"]
        == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )
    assert measurement["observer_id"] == "test-video-observer/0.1"
    assert measurement["production_measurement_evidence"] is True
    assert measurement["sample"] == {"width": 2, "height": 2, "frame_count": 3}
    assert measurement["metrics"]["identity_similarity"] == pytest.approx(0.94)
    assert measurement["metrics"]["motion_quality"] == pytest.approx(0.88)
    assert measurement["metrics"]["artifact_integrity"] == pytest.approx(1.0)
    assert measurement["metrics"]["temporal_consistency"] == pytest.approx(1.0)
    assert measurement["semantic_scorer"] == {
        "schema": "test-semantic-scorer/0.1",
        "origin": "external-pretrained-foundation",
    }


def test_video_observer_integrates_with_artifact_measured_quality_gate(tmp_path):
    artifact = tmp_path / "candidate.mp4"
    artifact.write_bytes(b"actual-rendered-video-container")
    observer = ArtifactVideoMetricObserver(AttestedSemanticScorer(), sampler=_sampler)
    evaluator = ArtifactMeasuredSequenceQualityEvaluator(observer)

    report = evaluator(str(artifact), shot=Shot(), attempt_index=0)

    assert report["production_measurement_evidence"] is True
    assert report["accepted"] is True
    assert report["metrics"]["anatomy_quality"] == pytest.approx(0.86)


def test_unattested_semantic_scorer_cannot_be_promoted_to_production_qc(tmp_path):
    artifact = tmp_path / "candidate.mp4"
    artifact.write_bytes(b"actual-rendered-video-container")
    observer = ArtifactVideoMetricObserver(_semantic_scorer, sampler=_sampler)

    measurement = observer(str(artifact), shot=Shot(), attempt_index=0)
    assert measurement["production_measurement_evidence"] is False

    with pytest.raises(TypeError, match="production_measurement_evidence=True"):
        ArtifactMeasuredSequenceQualityEvaluator(observer)


def test_video_observer_rejects_missing_semantic_identity_or_motion(tmp_path):
    artifact = tmp_path / "candidate.mp4"
    artifact.write_bytes(b"actual-rendered-video-container")
    observer = ArtifactVideoMetricObserver(
        lambda *_args, **_kwargs: {"identity_similarity": 0.9},
        sampler=_sampler,
    )

    with pytest.raises(VideoArtifactObservationError, match="motion_quality"):
        observer(str(artifact), shot=Shot(), attempt_index=0)


def test_video_observer_rejects_empty_artifact_before_sampling(tmp_path):
    artifact = tmp_path / "candidate.mp4"
    artifact.touch()
    observer = ArtifactVideoMetricObserver(AttestedSemanticScorer(), sampler=_sampler)

    with pytest.raises(VideoArtifactObservationError, match="missing or empty"):
        observer(str(artifact), shot=Shot(), attempt_index=0)


def test_video_observer_rejects_out_of_range_semantic_metric(tmp_path):
    artifact = tmp_path / "candidate.mp4"
    artifact.write_bytes(b"actual-rendered-video-container")
    observer = ArtifactVideoMetricObserver(
        lambda *_args, **_kwargs: {
            "identity_similarity": 1.2,
            "motion_quality": 0.9,
        },
        sampler=_sampler,
    )

    with pytest.raises(VideoArtifactObservationError, match="between 0 and 1"):
        observer(str(artifact), shot=Shot(), attempt_index=0)


@pytest.mark.parametrize(
    "observer_metric", ["artifact_integrity", "temporal_consistency"]
)
def test_semantic_scorer_cannot_override_observer_owned_metrics(
    tmp_path,
    observer_metric,
):
    artifact = tmp_path / "candidate.mp4"
    artifact.write_bytes(b"actual-rendered-video-container")
    observer = ArtifactVideoMetricObserver(
        lambda *_args, **_kwargs: {
            "identity_similarity": 0.94,
            "motion_quality": 0.88,
            observer_metric: 0.0,
        },
        sampler=_sampler,
    )

    with pytest.raises(
        VideoArtifactObservationError,
        match="cannot override observer-owned metric",
    ):
        observer(str(artifact), shot=Shot(), attempt_index=0)


def test_video_observer_rejects_invalid_semantic_runtime_provenance(tmp_path):
    artifact = tmp_path / "candidate.mp4"
    artifact.write_bytes(b"actual-rendered-video-container")

    class InvalidProvenanceScorer(AttestedSemanticScorer):
        runtime_provenance = "not-callable"

    observer = ArtifactVideoMetricObserver(
        InvalidProvenanceScorer(),
        sampler=_sampler,
    )
    with pytest.raises(VideoArtifactObservationError, match="must be callable"):
        observer(str(artifact), shot=Shot(), attempt_index=0)
