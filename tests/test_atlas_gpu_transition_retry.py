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
from cineos.atlas.transition_quality import TRANSITION_QUALITY_SCHEMA


def _request(index: int) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-transition",
        camera={"movement": "tracking"},
        characters=[{"character_id": "lead"}],
        environment={"location": "street"},
        wardrobe=[],
        props=[],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={"action": "walk"},
        approved_reference_ids=["lead-approved-reference"],
        deterministic_seed=7000 + index,
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
    request: NativeShotRequest,
    output_dir: Path,
) -> GPUFoundationExecutionReceipt:
    artifact = output_dir / (
        f"{request.scene_id}-{request.shot_id}-{request.content_hash[:8]}.mp4"
    )
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


def _digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _transition_report(
    previous_path: str,
    current_path: str,
    *,
    previous_shot,
    current_shot,
    accepted: bool,
) -> dict[str, object]:
    return {
        "schema": TRANSITION_QUALITY_SCHEMA,
        "production_measurement_evidence": True,
        "accepted": accepted,
        "observer_id": "terminal-initial-embedding-v1",
        "previous_scene_id": previous_shot.scene_id,
        "previous_shot_id": previous_shot.shot_id,
        "current_scene_id": current_shot.scene_id,
        "current_shot_id": current_shot.shot_id,
        "previous_output_sha256": _digest(previous_path),
        "current_output_sha256": _digest(current_path),
        "measured_sample_count": 4,
        "failed_metrics": [] if accepted else ["cross_shot_visual_seam"],
        "directives": (
            []
            if accepted
            else ["preserve predecessor pose, lighting, and spatial composition"]
        ),
    }


def test_failed_transition_rerenders_only_current_shot(tmp_path: Path) -> None:
    requests = [_request(index) for index in range(5)]
    rendered: list[NativeShotRequest] = []
    transition_calls: dict[str, int] = {}

    def executor(request, profile, *, output_dir):
        rendered.append(request)
        return _receipt(request, Path(output_dir))

    def quality_evaluator(_path, *, shot, attempt_index):
        return {"accepted": True, "failed_metrics": [], "directives": []}

    def transition_evaluator(
        previous_path,
        current_path,
        *,
        previous_shot,
        current_shot,
        attempt_index,
    ):
        count = transition_calls.get(current_shot.shot_id, 0)
        transition_calls[current_shot.shot_id] = count + 1
        accepted = current_shot.shot_id != "shot-2" or count > 0
        return _transition_report(
            previous_path,
            current_path,
            previous_shot=previous_shot,
            current_shot=current_shot,
            accepted=accepted,
        )

    receipt = run_quality_retry_connected_gpu_benchmark(
        "transition-retry",
        requests,
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        quality_evaluator=quality_evaluator,
        transition_evaluator=transition_evaluator,
        retry_policy=QualityRetryPolicy(max_attempts=3, seed_stride=73),
        shot_executor=executor,
    )

    payload = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
    gate = payload["quality_retry_gate"]
    retried = gate["shots"][2]

    assert len(rendered) == 6
    assert gate["transition_gate_applied"] is True
    assert gate["accepted_transition_count"] == 4
    assert retried["attempt_count"] == 2
    assert len(retried["transition_attempts"]) == 2
    assert retried["transition_attempts"][0]["accepted"] is False
    assert retried["accepted_transition"]["accepted"] is True
    assert rendered[3].metadata["quality_directives"] == [
        "preserve predecessor pose, lighting, and spatial composition"
    ]
    assert receipt.shot_receipts[1].result.request_hash == requests[1].content_hash


def test_tampered_transition_hash_fails_closed(tmp_path: Path) -> None:
    requests = [_request(index) for index in range(5)]

    def transition_evaluator(
        previous_path,
        current_path,
        *,
        previous_shot,
        current_shot,
        attempt_index,
    ):
        report = _transition_report(
            previous_path,
            current_path,
            previous_shot=previous_shot,
            current_shot=current_shot,
            accepted=True,
        )
        report["previous_output_sha256"] = "0" * 64
        return report

    with pytest.raises(GPUQualityRetryBenchmarkError, match="predecessor artifact"):
        run_quality_retry_connected_gpu_benchmark(
            "transition-tamper",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            quality_evaluator=lambda *_args, **_kwargs: {
                "accepted": True,
                "failed_metrics": [],
                "directives": [],
            },
            transition_evaluator=transition_evaluator,
            shot_executor=lambda request, profile, *, output_dir: _receipt(
                request, Path(output_dir)
            ),
        )

    assert not (tmp_path / "transition-tamper.gpu-quality-retry.json").exists()
