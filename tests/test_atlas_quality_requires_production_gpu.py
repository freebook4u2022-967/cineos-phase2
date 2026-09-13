import hashlib
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
        scene_id="scene-production-evidence",
        camera={"movement": "tracking"},
        characters=[{"character_id": "lead"}],
        environment={"location": "street"},
        wardrobe=[],
        props=[],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={"action": "walk"},
        approved_reference_ids=["lead-approved-reference"],
        deterministic_seed=4000 + index,
        renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
        metadata={"prompt": f"lead continues through shot {index}"},
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


def _injected_executor(request, profile, *, output_dir):
    artifact = Path(output_dir) / f"{request.scene_id}-{request.shot_id}.mp4"
    payload = f"injected-{request.shot_id}-{request.content_hash}".encode()
    artifact.write_bytes(payload)
    result = DiffusersVideoResult(
        shot_id=request.shot_id,
        scene_id=request.scene_id,
        output_path=str(artifact),
        frame_count=48,
        seed=request.deterministic_seed,
        foundation=profile.provenance,
        request_hash=request.content_hash,
    )
    return GPUFoundationExecutionReceipt(
        result=result,
        execution_plan=_plan(),
        profile_id=profile.profile_id,
        origin=profile.origin,
        output_bytes=len(payload),
        output_sha256=hashlib.sha256(payload).hexdigest(),
        elapsed_seconds=0.1,
    )


def _quality_evaluator():
    return CineosSequenceQualityEvaluator(
        lambda *_args, **_kwargs: {
            "identity_similarity": 0.95,
            "temporal_consistency": 0.93,
            "artifact_integrity": 0.99,
            "motion_quality": 0.91,
        }
    )


def test_production_requirement_rejects_injected_executor_and_removes_manifest(
    tmp_path,
):
    benchmark_id = "production-required"

    with pytest.raises(
        GPUQualityBenchmarkError, match="production GPU evidence required"
    ):
        run_quality_gated_connected_gpu_benchmark(
            benchmark_id,
            [_request(index) for index in range(5)],
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            quality_evaluator=_quality_evaluator(),
            shot_executor=_injected_executor,
            require_production_gpu=True,
        )

    assert not (tmp_path / f"{benchmark_id}.gpu-benchmark.json").exists()


def test_default_compatibility_still_allows_injected_regression_benchmark(tmp_path):
    receipt = run_quality_gated_connected_gpu_benchmark(
        "regression-compatible",
        [_request(index) for index in range(5)],
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        quality_evaluator=_quality_evaluator(),
        shot_executor=_injected_executor,
    )

    assert receipt.production_gpu_evidence is False
    assert receipt.evidence_tier == "non-production-or-injected"
    assert Path(receipt.manifest_path).exists()


def test_production_requirement_must_be_boolean(tmp_path):
    with pytest.raises(TypeError, match="require_production_gpu must be a bool"):
        run_quality_gated_connected_gpu_benchmark(
            "invalid-production-flag",
            [_request(index) for index in range(5)],
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            quality_evaluator=_quality_evaluator(),
            shot_executor=_injected_executor,
            require_production_gpu="yes",
        )
