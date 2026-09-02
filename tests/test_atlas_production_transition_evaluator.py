import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cineos.atlas.gpu_production_quality_retry import (
    ProductionGPUQualityRetryError,
    run_production_continuity_quality_retry_connected_gpu_benchmark,
)
from cineos.atlas.production_continuity_diffusers import VISUAL_CONTINUITY_SCHEMA
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
    assert (
        report["previous_output_sha256"]
        == hashlib.sha256(previous.read_bytes()).hexdigest()
    )
    assert (
        report["current_output_sha256"]
        == hashlib.sha256(current.read_bytes()).hexdigest()
    )


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


def _connected_receipts(*, omit_lineage_at: int | None = None):
    receipts = []
    for index in range(5):
        artifact_sha = f"{index + 1:064x}"
        request_hash = f"request-{index}"
        if index == 0:
            provenance = {
                "schema": VISUAL_CONTINUITY_SCHEMA,
                "mode": "approved_reference_root",
                "scene_id": "s1",
                "shot_id": "q0",
                "current_artifact_sha256": artifact_sha,
                "current_request_hash": request_hash,
                "previous_scene_id": None,
                "previous_shot_id": None,
                "predecessor_artifact_sha256": None,
                "predecessor_request_hash": None,
                "in_memory_terminal_frame": False,
            }
        else:
            provenance = {
                "schema": VISUAL_CONTINUITY_SCHEMA,
                "mode": "predecessor_terminal_frame_lineage",
                "scene_id": "s1",
                "shot_id": f"q{index}",
                "current_artifact_sha256": artifact_sha,
                "current_request_hash": request_hash,
                "previous_scene_id": "s1",
                "previous_shot_id": f"q{index - 1}",
                "predecessor_artifact_sha256": f"{index:064x}",
                "predecessor_request_hash": f"request-{index - 1}",
                "in_memory_terminal_frame": True,
            }
        if index == omit_lineage_at:
            provenance = None
        receipts.append(
            SimpleNamespace(
                output_sha256=artifact_sha,
                result=SimpleNamespace(
                    scene_id="s1",
                    shot_id=f"q{index}",
                    request_hash=request_hash,
                    conditioning_provenance=provenance,
                ),
            )
        )
    return tuple(receipts)


def _strict_receipt(tmp_path: Path, *, omit_lineage_at: int | None = None):
    manifest = tmp_path / "benchmark.json"
    manifest.write_text(
        json.dumps(
            {
                "quality_retry_gate": {
                    "transition_gate_applied": True,
                    "accepted_transition_count": 4,
                    "accepted_transitions": [
                        {"production_measurement_evidence": True} for _ in range(4)
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        manifest_path=str(manifest),
        shot_receipts=_connected_receipts(omit_lineage_at=omit_lineage_at),
    )


def test_strict_production_entry_requires_terminal_frame_lineage(
    monkeypatch, tmp_path: Path
) -> None:
    aggregate = _strict_receipt(tmp_path, omit_lineage_at=2)
    monkeypatch.setattr(
        "cineos.atlas.gpu_production_quality_retry."
        "run_production_quality_retry_connected_gpu_benchmark",
        lambda *_args, **_kwargs: aggregate,
    )

    with pytest.raises(
        ProductionGPUQualityRetryError,
        match="artifact-bound terminal-frame lineage",
    ):
        run_production_continuity_quality_retry_connected_gpu_benchmark(
            "lineage-required",
            [SimpleNamespace()] * 5,
            SimpleNamespace(),
            output_dir=tmp_path,
            quality_evaluator=SimpleNamespace(),
            transition_evaluator=ArtifactMeasuredTransitionQualityEvaluator(
                MeasuredObserver()
            ),
        )

    assert not Path(aggregate.manifest_path).exists()


def test_strict_production_entry_accepts_complete_lineage_and_transition_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    aggregate = _strict_receipt(tmp_path)
    monkeypatch.setattr(
        "cineos.atlas.gpu_production_quality_retry."
        "run_production_quality_retry_connected_gpu_benchmark",
        lambda *_args, **_kwargs: aggregate,
    )

    result = run_production_continuity_quality_retry_connected_gpu_benchmark(
        "lineage-complete",
        [SimpleNamespace()] * 5,
        SimpleNamespace(),
        output_dir=tmp_path,
        quality_evaluator=SimpleNamespace(),
        transition_evaluator=ArtifactMeasuredTransitionQualityEvaluator(
            MeasuredObserver()
        ),
    )

    assert result is aggregate
    assert Path(aggregate.manifest_path).exists()
