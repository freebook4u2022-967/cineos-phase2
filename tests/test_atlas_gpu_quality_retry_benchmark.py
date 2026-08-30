import hashlib
import json
from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoResult
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_foundation_smoke import GPUFoundationExecutionReceipt
from cineos.atlas.gpu_preflight import GPUExecutionPlan
from cineos.atlas.gpu_quality_retry_benchmark import (
    GPUQualityRetryBenchmarkError,
    run_quality_retry_connected_gpu_benchmark,
)
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.quality_retry import QualityRetryPolicy


def _request(index: int) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-retry",
        camera={"movement": "tracking"},
        characters=[{"character_id": "lead"}],
        environment={"location": "street"},
        wardrobe=[],
        props=[],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={"action": "walk"},
        approved_reference_ids=["lead-approved-reference"],
        deterministic_seed=3000 + index,
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
    payload = f"video-{request.shot_id}-{request.content_hash}".encode()
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
        elapsed_seconds=0.2,
    )


def test_rejected_shot_is_rerendered_with_fresh_hash_seed_and_lineage(tmp_path):
    requests = [_request(index) for index in range(5)]
    rendered = []

    def executor(request, profile, *, output_dir):
        rendered.append(request)
        return _receipt(request, Path(output_dir))

    def evaluator(_path, *, shot, attempt_index):
        if shot.shot_id == "shot-2" and attempt_index == 0:
            return {
                "accepted": False,
                "failed_metrics": ["identity_similarity"],
                "directives": [
                    "preserve approved character identity and facial structure"
                ],
            }
        return {
            "accepted": True,
            "failed_metrics": [],
            "directives": [],
        }

    receipt = run_quality_retry_connected_gpu_benchmark(
        "retry-pass",
        requests,
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        quality_evaluator=evaluator,
        retry_policy=QualityRetryPolicy(max_attempts=3, seed_stride=101),
        shot_executor=executor,
    )

    payload = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
    gate = payload["quality_retry_gate"]
    retried = gate["shots"][2]
    original = requests[2]
    accepted = receipt.shot_receipts[2]

    assert gate["accepted"] is True
    assert gate["shot_count"] == 5
    assert retried["attempt_count"] == 2
    assert retried["original_request_hash"] == original.content_hash
    assert retried["accepted_request_hash"] != original.content_hash
    assert accepted.result.seed == original.deterministic_seed + 101
    assert (
        retried["attempts"][1]["effective_request_hash"] == accepted.result.request_hash
    )
    assert (
        rendered[3].metadata["quality_retry"]["parent_request_hash"]
        == original.content_hash
    )
    assert rendered[3].metadata["quality_directives"] == [
        "preserve approved character identity and facial structure"
    ]


def test_retry_exhaustion_fails_closed_without_completed_manifest(tmp_path):
    requests = [_request(index) for index in range(5)]

    def evaluator(_path, *, shot, attempt_index):
        return {
            "accepted": False,
            "failed_metrics": ["temporal_consistency"],
            "directives": ["reduce cross-frame and cross-shot temporal drift"],
        }

    with pytest.raises(GPUQualityRetryBenchmarkError, match="retry exhausted"):
        run_quality_retry_connected_gpu_benchmark(
            "retry-fail",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            quality_evaluator=evaluator,
            retry_policy=QualityRetryPolicy(max_attempts=2, seed_stride=17),
            shot_executor=lambda request, profile, *, output_dir: _receipt(
                request, Path(output_dir)
            ),
        )

    assert not (tmp_path / "retry-fail.gpu-quality-retry.json").exists()


def test_retry_receipt_must_bind_effective_request_hash(tmp_path):
    requests = [_request(index) for index in range(5)]

    def bad_executor(request, profile, *, output_dir):
        receipt = _receipt(request, Path(output_dir))
        wrong_result = DiffusersVideoResult(
            shot_id=receipt.result.shot_id,
            scene_id=receipt.result.scene_id,
            output_path=receipt.result.output_path,
            frame_count=receipt.result.frame_count,
            seed=receipt.result.seed,
            foundation=receipt.result.foundation,
            request_hash="0" * 64,
        )
        return GPUFoundationExecutionReceipt(
            result=wrong_result,
            execution_plan=receipt.execution_plan,
            profile_id=receipt.profile_id,
            origin=receipt.origin,
            output_bytes=receipt.output_bytes,
            output_sha256=receipt.output_sha256,
            elapsed_seconds=receipt.elapsed_seconds,
        )

    with pytest.raises(
        GPUQualityRetryBenchmarkError, match="effective rerender request"
    ):
        run_quality_retry_connected_gpu_benchmark(
            "retry-bad-hash",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            quality_evaluator=lambda *_args, **_kwargs: {
                "accepted": True,
                "failed_metrics": [],
                "directives": [],
            },
            shot_executor=bad_executor,
        )

    assert not (tmp_path / "retry-bad-hash.gpu-quality-retry.json").exists()
