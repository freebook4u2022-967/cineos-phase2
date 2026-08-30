import hashlib
import json
from pathlib import Path

from cineos.atlas.diffusers_video import DiffusersVideoResult
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_foundation_smoke import GPUFoundationExecutionReceipt
from cineos.atlas.gpu_preflight import GPUExecutionPlan
from cineos.atlas.gpu_quality_benchmark import run_quality_gated_connected_gpu_benchmark
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.sequence_quality import CineosSequenceQualityEvaluator


def _request(index: int) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-rerender",
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
        metadata={"prompt": f"lead walks through connected shot {index}"},
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


def test_rejected_shot_is_correctively_rerendered_and_hash_bound(tmp_path):
    requests = [_request(index) for index in range(5)]
    original_hash = requests[2].content_hash
    original_seed = requests[2].deterministic_seed
    executions = []

    def executor(request, profile, *, output_dir):
        executions.append(
            (
                request.shot_id,
                request.deterministic_seed,
                request.content_hash,
                request.metadata.get("prompt"),
            )
        )
        artifact = Path(output_dir) / f"{request.scene_id}-{request.shot_id}.mp4"
        payload = (
            f"video-{request.shot_id}-{request.deterministic_seed}-{request.content_hash}"
        ).encode()
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
            elapsed_seconds=0.25,
        )

    def metrics(_path, *, shot, attempt_index):
        identity = 0.40 if shot.shot_id == "shot-2" and attempt_index == 0 else 0.95
        return {
            "identity_similarity": identity,
            "temporal_consistency": 0.92,
            "artifact_integrity": 0.99,
            "motion_quality": 0.90,
        }

    receipt = run_quality_gated_connected_gpu_benchmark(
        "quality-rerender",
        requests,
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        quality_evaluator=CineosSequenceQualityEvaluator(metrics),
        shot_executor=executor,
        max_quality_attempts=2,
        retry_seed_stride=101,
    )

    shot2_calls = [item for item in executions if item[0] == "shot-2"]
    assert len(shot2_calls) == 2
    assert shot2_calls[0][1] == original_seed
    assert shot2_calls[1][1] == original_seed + 101
    assert shot2_calls[0][2] == original_hash
    assert shot2_calls[1][2] != original_hash
    assert "Quality correction:" in shot2_calls[1][3]
    assert "preserve approved character identity" in shot2_calls[1][3]

    assert requests[2].content_hash == shot2_calls[1][2]
    assert requests[2].metadata["quality_retry"]["initial_request_hash"] == original_hash

    report = receipt.quality_reports[2]
    assert report["accepted"] is True
    assert report["rerendered"] is True
    assert report["attempt_count"] == 2
    assert report["initial_request_hash"] == original_hash
    assert [attempt["accepted"] for attempt in report["attempts"]] == [False, True]
    assert report["request_hash"] == requests[2].content_hash

    payload = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
    assert payload["quality_gate"]["rerendered_shot_count"] == 1
    assert payload["quality_gate"]["reports"][2]["attempt_count"] == 2
    assert payload["shots"][2]["request_hash"] == requests[2].content_hash
