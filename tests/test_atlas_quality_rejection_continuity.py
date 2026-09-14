import hashlib
from pathlib import Path

from cineos.atlas.diffusers_video import DiffusersVideoResult
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_foundation_smoke import GPUFoundationExecutionReceipt
from cineos.atlas.gpu_preflight import GPUExecutionPlan
from cineos.atlas.gpu_quality_retry_benchmark import (
    run_quality_retry_connected_gpu_benchmark,
)
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.quality_retry import QualityRetryPolicy


def _request(index: int) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-rejection-continuity",
        camera={"movement": "tracking"},
        characters=[{"character_id": "lead"}],
        environment={"location": "street"},
        wardrobe=[],
        props=[],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={"action": "walk"},
        approved_reference_ids=["lead-approved-reference"],
        deterministic_seed=4100 + index,
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


def _receipt(request: NativeShotRequest, output_dir: Path):
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
        elapsed_seconds=0.1,
    )


class RecordingExecutor:
    def __init__(self):
        self.rendered = []
        self.discarded = []

    def __call__(self, request, profile, *, output_dir):
        self.rendered.append(request)
        return _receipt(request, Path(output_dir))

    def discard_quality_rejected_result(self, receipt):
        self.discarded.append(receipt)


def test_rejected_attempt_is_discarded_before_retry_and_only_rejected_attempt(tmp_path):
    requests = [_request(index) for index in range(5)]
    executor = RecordingExecutor()

    def evaluator(_path, *, shot, attempt_index):
        rejected = shot.shot_id == "shot-2" and attempt_index == 0
        return {
            "accepted": not rejected,
            "failed_metrics": ["identity_similarity"] if rejected else [],
            "directives": ["preserve identity"] if rejected else [],
        }

    receipt = run_quality_retry_connected_gpu_benchmark(
        "rejection-continuity",
        requests,
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        quality_evaluator=evaluator,
        retry_policy=QualityRetryPolicy(max_attempts=2, seed_stride=19),
        shot_executor=executor,
    )

    assert len(executor.discarded) == 1
    rejected = executor.discarded[0]
    assert rejected.result.shot_id == "shot-2"
    assert rejected.result.request_hash == requests[2].content_hash
    assert receipt.shot_receipts[2].result.request_hash != rejected.result.request_hash
    assert len(executor.rendered) == 6
