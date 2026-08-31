import hashlib

from cineos.atlas.artifact_video_observer import ArtifactVideoMetricObserver, RGBVideoSample
from cineos.atlas.semantic_video_scorer import LearnedIdentityMotionScorer
from cineos.atlas.sequence_quality import ArtifactMeasuredSequenceQualityEvaluator


class Shot:
    approved_reference_ids = ["lead-approved"]


def test_learned_semantic_scores_flow_into_artifact_bound_quality_report(tmp_path):
    artifact = tmp_path / "generated.mp4"
    artifact.write_bytes(b"real-rendered-video-evidence")
    frame = bytes([40, 80, 120] * 4)
    sample = RGBVideoSample(2, 2, (frame, frame, frame))

    semantic = LearnedIdentityMotionScorer(
        lambda _sample: [(1.0, 0.0)] * len(_sample.frames),
        lambda reference_id: (
            (1.0, 0.0) if reference_id == "lead-approved" else (0.0, 1.0)
        ),
        lambda *_args, **_kwargs: 0.95,
    )
    observer = ArtifactVideoMetricObserver(
        semantic,
        sampler=lambda _artifact: sample,
        observer_id="cineos-learned-semantic-test/0.1",
    )
    evaluator = ArtifactMeasuredSequenceQualityEvaluator(observer)

    report = evaluator(
        str(artifact),
        shot=Shot(),
        attempt_index=0,
    )

    assert report["accepted"] is True
    assert report["production_measurement_evidence"] is True
    assert report["metrics"]["identity_similarity"] == 1.0
    assert report["metrics"]["motion_quality"] == 0.95
    assert report["measurement"]["artifact_sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
