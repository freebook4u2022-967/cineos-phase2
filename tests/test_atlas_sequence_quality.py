import hashlib

import pytest

from cineos.atlas.sequence_quality import (
    PRODUCTION_MEASUREMENT_SCHEMA,
    ArtifactMeasuredSequenceQualityEvaluator,
    CineosSequenceQualityEvaluator,
    SequenceQualityError,
    SequenceQualityPolicy,
)


class Shot:
    shot_id = "shot-01"


class AttestedTestObserver:
    production_measurement_evidence = True
    observer_id = "test-artifact-observer/0.1"

    def __init__(self, artifact, metrics, *, measured_sha256=None, reported_observer_id=None):
        self.artifact = artifact
        self.metrics = metrics
        self.measured_sha256 = measured_sha256
        self.reported_observer_id = reported_observer_id

    def __call__(self, *_args, **_kwargs):
        digest = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
        return {
            "schema": PRODUCTION_MEASUREMENT_SCHEMA,
            "observer_id": self.reported_observer_id or self.observer_id,
            "artifact_sha256": self.measured_sha256 or digest,
            "metrics": self.metrics,
        }


def _evaluate(metrics, *, policy=None):
    evaluator = CineosSequenceQualityEvaluator(
        lambda *_args, **_kwargs: metrics,
        policy=policy,
    )
    return evaluator("candidate.mp4", shot=Shot(), attempt_index=0)


def _measured_evaluator(
    artifact, metrics, *, measured_sha256=None, reported_observer_id=None
):
    return ArtifactMeasuredSequenceQualityEvaluator(
        AttestedTestObserver(
            artifact,
            metrics,
            measured_sha256=measured_sha256,
            reported_observer_id=reported_observer_id,
        )
    )


def test_accepts_shot_when_core_metrics_and_overall_score_pass():
    report = _evaluate(
        {
            "identity_similarity": 0.93,
            "temporal_consistency": 0.91,
            "artifact_integrity": 1.0,
            "motion_quality": 0.88,
            "anatomy_quality": 0.86,
        }
    )

    assert report["accepted"] is True
    assert report["decision"] == "accept"
    assert report["failed_metrics"] == []
    assert report["score"] >= report["policy"]["overall_floor"]


def test_hard_identity_floor_rejects_even_if_other_metrics_are_high():
    report = _evaluate(
        {
            "identity_similarity": 0.70,
            "temporal_consistency": 0.99,
            "artifact_integrity": 1.0,
            "motion_quality": 0.99,
        }
    )

    assert report["accepted"] is False
    assert "identity_similarity" in report["failed_metrics"]
    assert any("identity" in item for item in report["directives"])


def test_optional_difficult_case_metrics_contribute_to_overall_score():
    policy = SequenceQualityPolicy(overall_floor=0.90)
    base = {
        "identity_similarity": 0.94,
        "temporal_consistency": 0.94,
        "artifact_integrity": 0.98,
        "motion_quality": 0.94,
    }
    strong = _evaluate(
        {
            **base,
            "anatomy_quality": 0.98,
            "object_interaction_quality": 0.98,
            "dialogue_lip_sync": 0.98,
        },
        policy=policy,
    )
    weak = _evaluate(
        {
            **base,
            "anatomy_quality": 0.10,
            "object_interaction_quality": 0.10,
            "dialogue_lip_sync": 0.10,
        },
        policy=policy,
    )

    assert strong["score"] > weak["score"]
    assert strong["accepted"] is True
    assert weak["accepted"] is False
    assert "overall_score" in weak["failed_metrics"]


def test_missing_core_metric_fails_closed():
    with pytest.raises(SequenceQualityError, match="motion_quality"):
        _evaluate(
            {
                "identity_similarity": 0.9,
                "temporal_consistency": 0.9,
                "artifact_integrity": 1.0,
            }
        )


def test_metric_outside_normalized_range_fails_closed():
    with pytest.raises(SequenceQualityError, match="between 0 and 1"):
        _evaluate(
            {
                "identity_similarity": 1.2,
                "temporal_consistency": 0.9,
                "artifact_integrity": 1.0,
                "motion_quality": 0.9,
            }
        )


def test_artifact_measured_evaluator_binds_report_to_exact_rendered_bytes(tmp_path):
    artifact = tmp_path / "candidate.mp4"
    artifact.write_bytes(b"real-rendered-video-bytes")
    evaluator = _measured_evaluator(
        artifact,
        {
            "identity_similarity": 0.95,
            "temporal_consistency": 0.93,
            "artifact_integrity": 0.99,
            "motion_quality": 0.91,
        },
    )

    report = evaluator(str(artifact), shot=Shot(), attempt_index=0)

    assert report["accepted"] is True
    assert report["production_measurement_evidence"] is True
    assert report["measurement"]["observer_id"] == "test-artifact-observer/0.1"
    assert report["measurement"]["observer_attested"] is True
    assert (
        report["measurement"]["artifact_sha256"]
        == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )


def test_artifact_measured_evaluator_rejects_stale_or_foreign_digest(tmp_path):
    artifact = tmp_path / "candidate.mp4"
    artifact.write_bytes(b"current-render")
    evaluator = _measured_evaluator(
        artifact,
        {
            "identity_similarity": 0.95,
            "temporal_consistency": 0.93,
            "artifact_integrity": 0.99,
            "motion_quality": 0.91,
        },
        measured_sha256="0" * 64,
    )

    with pytest.raises(SequenceQualityError, match="SHA-256"):
        evaluator(str(artifact), shot=Shot(), attempt_index=0)


def test_artifact_measured_evaluator_rejects_unattested_callable():
    with pytest.raises(TypeError, match="attest production_measurement_evidence"):
        ArtifactMeasuredSequenceQualityEvaluator(lambda *_args, **_kwargs: {})


def test_artifact_measured_evaluator_rejects_observer_identity_spoof(tmp_path):
    artifact = tmp_path / "candidate.mp4"
    artifact.write_bytes(b"current-render")
    evaluator = _measured_evaluator(
        artifact,
        {
            "identity_similarity": 0.95,
            "temporal_consistency": 0.93,
            "artifact_integrity": 0.99,
            "motion_quality": 0.91,
        },
        reported_observer_id="spoofed-observer/9.9",
    )

    with pytest.raises(SequenceQualityError, match="attested observer"):
        evaluator(str(artifact), shot=Shot(), attempt_index=0)


def test_artifact_measured_evaluator_requires_real_artifact(tmp_path):
    observer = AttestedTestObserver(
        tmp_path / "missing.mp4",
        {
            "identity_similarity": 0.95,
            "temporal_consistency": 0.93,
            "artifact_integrity": 0.99,
            "motion_quality": 0.91,
        },
    )
    evaluator = ArtifactMeasuredSequenceQualityEvaluator(observer)

    with pytest.raises(SequenceQualityError, match="does not exist"):
        evaluator(str(tmp_path / "missing.mp4"), shot=Shot(), attempt_index=0)


def test_policy_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="overall_floor"):
        SequenceQualityPolicy(overall_floor=1.1)
