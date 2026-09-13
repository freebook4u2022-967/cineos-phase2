import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoResult
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_connected_benchmark import (
    GPUConnectedBenchmarkError,
    run_connected_gpu_benchmark,
)
from cineos.atlas.gpu_foundation_smoke import GPUFoundationExecutionReceipt
from cineos.atlas.gpu_preflight import GPUExecutionPlan
from cineos.atlas.native_request import NativeShotRequest


def _request(index: int) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-evidence",
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


def _receipt(
    request: NativeShotRequest, output_dir: Path
) -> GPUFoundationExecutionReceipt:
    artifact = output_dir / f"{request.scene_id}-{request.shot_id}.mp4"
    payload = f"video-{request.shot_id}".encode()
    artifact.write_bytes(payload)
    return GPUFoundationExecutionReceipt(
        result=DiffusersVideoResult(
            shot_id=request.shot_id,
            scene_id=request.scene_id,
            output_path=str(artifact),
            frame_count=48,
            seed=request.deterministic_seed,
            foundation=WAN22_TI2V_5B_PROFILE.provenance,
            request_hash=request.content_hash,
        ),
        execution_plan=_plan(),
        profile_id=WAN22_TI2V_5B_PROFILE.profile_id,
        origin=WAN22_TI2V_5B_PROFILE.origin,
        output_bytes=len(payload),
        output_sha256=hashlib.sha256(payload).hexdigest(),
        elapsed_seconds=0.25,
    )


def test_connected_benchmark_rejects_duplicate_video_payloads(tmp_path):
    requests = [_request(index) for index in range(5)]
    duplicate_hash = hashlib.sha256(b"recycled-video").hexdigest()

    def executor(request, profile, *, output_dir):
        assert profile is WAN22_TI2V_5B_PROFILE
        receipt = _receipt(request, Path(output_dir))
        return replace(receipt, output_sha256=duplicate_hash)

    with pytest.raises(GPUConnectedBenchmarkError, match="duplicate video payloads"):
        run_connected_gpu_benchmark(
            "duplicate-payload",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=executor,
        )

    assert not (tmp_path / "duplicate-payload.gpu-benchmark.json").exists()


def test_connected_benchmark_rejects_reused_output_artifact_path(tmp_path):
    requests = [_request(index) for index in range(5)]
    shared_path = tmp_path / "recycled.mp4"

    def executor(request, profile, *, output_dir):
        assert profile is WAN22_TI2V_5B_PROFILE
        receipt = _receipt(request, Path(output_dir))
        result = replace(receipt.result, output_path=str(shared_path))
        return replace(receipt, result=result)

    with pytest.raises(GPUConnectedBenchmarkError, match="reused one output artifact"):
        run_connected_gpu_benchmark(
            "duplicate-path",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=executor,
        )

    assert not (tmp_path / "duplicate-path.gpu-benchmark.json").exists()
