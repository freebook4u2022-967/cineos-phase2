import hashlib
import json
from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoResult
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_foundation_smoke import GPUFoundationExecutionReceipt
from cineos.atlas.gpu_preflight import GPUExecutionPlan
from cineos.atlas.gpu_quality_benchmark import (
    GPUQualityBenchmarkError,
    run_quality_gated_connected_gpu_benchmark,
)
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.sequence_quality import CineosSequenceQualityEvaluator


def _request(index: int) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-quality",
        camera={"movement": "tracking"},
        characters=[{"character_id": "lead"}],
        environment={"location": "street"},
        wardrobe=[],
        props=[],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={"action": "walk"},
        approved_reference_ids=["lead-approved-reference"],
        deterministic_seed=2000 + index,
        renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
    )
    request.refresh_hash()
    return request


def _plan() -> GPUExecutionPlan:
    return GPUExecutionPlan(
        device="cuda:0",
        dtype="bfloat16",
        memory_strategy="resident",
        enable_vae_tiling=False,
        enable_vae_slicing=False,
        enable_attention_slicing=False,
        estimated_model_vram_gb=24.0,
        observed_total_vram_gb=48.0,
        observed_free_vram_gb=40.0,
        fit_margin_gb=16.0,
    )


def _receipt(request: NativeShotRequest, output_dir: Path) -> GPUFoundationExecutionReceipt:
    artifact = output_dir / f"{request.scene_id}-{request.shot_id}.mp4"
    payload = f"quality-video-{request.shot_id}".encode()
    artifact.write_bytes(payload)
    result = DiffusersVideoResult(
        shot_id=request.shot_id,
        scene_id=request.scene_id,
        output_path=str(artifact),
        frame_count=48,
        seed=request.deterministic_seed,
        foundation=WAN22_TI2V_5B_PROFILE.provenance,
        request_hash=request.content_hash,
    )
    return GPUFoundationExecutionReceipt(
        result=result,
        execution_plan=_plan(),
        profile_id=WAN22_TI2V_5B_PROFILE.profile_id,
        origin=WAN22_TI2V_5B_PROFILE.origin,
        output_bytes=len(payload),
        output_sha256=hashlib.sha256(payload).hexdigest(),
        elapsed_seconds=0.25,
    )


def _passing_evaluator():
    return CineosSequenceQualityEvaluator(
        lambda *_args, **_kwargs: {
            "identity_similarity": 0.94,
            "temporal_consistency": 0.91,
            "artifact_integrity": 0.99,
            "motion_quality": 0.88,
            "anatomy_quality": 0.85,
        }
    )


def test_quality_gated_connected_benchmark_persists_hash_bound_quality_evidence(tmp_path):
    requests = [_request(index) for index in range(5)]

    receipt = run_quality_gated_connected_gpu_benchmark(
        "quality-pass",
        requests,
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        quality_evaluator=_passing_evaluator(),
        shot_executor=lambda request, profile, *, output_dir: _receipt(
            request, Path(output_dir)
        ),
    )

    payload = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
    gate = payload["quality_gate"]
    assert gate["schema"] == "cineos-gpu-connected-quality-gate/0.1"
    assert gate["accepted"] is True
    assert gate["shot_count"] == 5
    assert [report["request_hash"] for report in gate["reports"]] == [
        request.content_hash for request in requests
    ]
    assert [report["output_sha256"] for report in gate["reports"]] == [
        shot["output_sha256"] for shot in payload["shots"]
    ]
    assert all(report["accepted"] is True for report in gate["reports"])


def test_quality_rejection_aborts_sequence_and_leaves_no_completed_manifest(tmp_path):
    requests = [_request(index) for index in range(5)]

    def metrics(_path, *, shot, attempt_index):
        assert attempt_index == 0
        identity = 0.40 if shot.shot_id == "shot-2" else 0.95
        return {
            "identity_similarity": identity,
            "temporal_consistency": 0.90,
            "artifact_integrity": 0.99,
            "motion_quality": 0.88,
        }

    evaluator = CineosSequenceQualityEvaluator(metrics)

    with pytest.raises(GPUQualityBenchmarkError, match="identity_similarity"):
        run_quality_gated_connected_gpu_benchmark(
            "quality-reject",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            quality_evaluator=evaluator,
            shot_executor=lambda request, profile, *, output_dir: _receipt(
                request, Path(output_dir)
            ),
        )

    assert not (tmp_path / "quality-reject.gpu-benchmark.json").exists()
    assert not (tmp_path / "scene-quality-shot-3.mp4").exists()


def test_quality_evaluator_must_return_a_report_mapping(tmp_path):
    requests = [_request(index) for index in range(5)]

    with pytest.raises(GPUQualityBenchmarkError, match="must return a dict report"):
        run_quality_gated_connected_gpu_benchmark(
            "bad-evaluator",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            quality_evaluator=lambda *_args, **_kwargs: None,
            shot_executor=lambda request, profile, *, output_dir: _receipt(
                request, Path(output_dir)
            ),
        )

    assert not (tmp_path / "bad-evaluator.gpu-benchmark.json").exists()
