import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from cineos.atlas.gpu_production_quality_retry import (
    ProductionGPUQualityRetryError,
    run_production_continuity_quality_retry_connected_gpu_benchmark,
)
from cineos.atlas.transition_quality import (
    TRANSITION_QUALITY_SCHEMA,
    ArtifactMeasuredTransitionQualityEvaluator,
    TransitionQualityError,
)


class MeasuredObserver:
    production_measurement_evidence = True
    observer_id = "learned-transition-observer-v1"

    def __init__(self, visual: float = 0.9, motion: float = 0.85) -> None:
        self.visual = visual
        self.motion = motion

    def __call__(
        self,
        previous_path,
        current_path,
        *,
        previous_shot,
        current_shot,
        attempt_index,
    ):
        return {
            "schema": TRANSITION_QUALITY_SCHEMA,
            "observer_id": self.observer_id,
            "previous_output_sha256": hashlib.sha256(
                Path(previous_path).read_bytes()
            ).hexdigest(),
            "current_output_sha256": hashlib.sha256(
                Path(current_path).read_bytes()
            ).hexdigest(),
            "measured_sample_count": 6,
            "metrics": {
                "visual_seam_similarity": self.visual,
                "motion_boundary_consistency": self.motion,
            },
        }


def test_plain_callable_cannot_be_promoted_to_production_transition_evidence() -> None:
    with pytest.raises(TypeError, match="attest"):
        ArtifactMeasuredTransitionQualityEvaluator(lambda *_args, **_kwargs: {})


def test_attested_evaluator_measures_and_binds_both_artifacts(tmp_path: Path) -> None:
    previous = tmp_path / "previous.mp4"
    current = tmp_path / "current.mp4"
    previous.write_bytes(b"previous-real-video")
    current.write_bytes(b"current-real-video")
    evaluator = ArtifactMeasuredTransitionQualityEvaluator(MeasuredObserver())

    report = evaluator(
        str(previous),
        str(current),
        previous_shot=SimpleNamespace(scene_id="s1", shot_id="q1"),
        current_shot=SimpleNamespace(scene_id="s1", shot_id="q2"),
        attempt_index=0,
    )

    assert report["accepted"] is True
    assert report["production_measurement_evidence"] is True
    assert report["measured_sample_count"] == 6
    assert report["previous_output_sha256"] == hashlib.sha256(
        previous.read_bytes()
    ).hexdigest()
    assert report["current_output_sha256"] == hashlib.sha256(
        current.read_bytes()
    ).hexdigest()


def test_attested_evaluator_generates_rerender_directives_for_bad_seam(
    tmp_path: Path,
) -> None:
    previous = tmp_path / "previous.mp4"
    current = tmp_path / "current.mp4"
    previous.write_bytes(b"previous-real-video")
    current.write_bytes(b"current-real-video")
    evaluator = ArtifactMeasuredTransitionQualityEvaluator(
        MeasuredObserver(visual=0.4, motion=0.5)
    )

    report = evaluator(
        str(previous),
        str(current),
        previous_shot=SimpleNamespace(scene_id="s1", shot_id="q1"),
        current_shot=SimpleNamespace(scene_id="s1", shot_id="q2"),
        attempt_index=0,
    )

    assert report["accepted"] is False
    assert report["failed_metrics"] == [
        "visual_seam_similarity",
        "motion_boundary_consistency",
    ]
    assert len(report["directives"]) == 2


def test_observer_hash_substitution_is_rejected(tmp_path: Path) -> None:
    class TamperedObserver(MeasuredObserver):
        def __call__(self, *args, **kwargs):
            report = super().__call__(*args, **kwargs)
            report["current_output_sha256"] = "0" * 64
            return report

    previous = tmp_path / "previous.mp4"
    current = tmp_path / "current.mp4"
    previous.write_bytes(b"previous")
    current.write_bytes(b"current")
    evaluator = ArtifactMeasuredTransitionQualityEvaluator(TamperedObserver())

    with pytest.raises(TransitionQualityError, match="current hash mismatch"):
        evaluator(
            str(previous),
            str(current),
            previous_shot=SimpleNamespace(scene_id="s1", shot_id="q1"),
            current_shot=SimpleNamespace(scene_id="s1", shot_id="q2"),
            attempt_index=0,
        )


def test_strict_production_entry_rejects_unattested_transition_before_gpu() -> None:
    with pytest.raises(ProductionGPUQualityRetryError, match="attested transition"):
        run_production_continuity_quality_retry_connected_gpu_benchmark(
            "must-not-render",
            [],
            SimpleNamespace(),
            output_dir="unused",
            quality_evaluator=SimpleNamespace(),
            transition_evaluator=lambda *_args, **_kwargs: {},
        )
