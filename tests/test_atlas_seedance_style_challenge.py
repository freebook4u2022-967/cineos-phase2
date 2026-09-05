import hashlib
import json
from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoResult
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_connected_benchmark import GPUConnectedBenchmarkError
from cineos.atlas.gpu_foundation_smoke import GPUFoundationExecutionReceipt
from cineos.atlas.gpu_preflight import GPUExecutionPlan
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.seedance_style_challenge import (
    REQUIRED_CHALLENGES,
    SeedanceStyleChallengeError,
    run_seedance_style_gpu_benchmark,
    validate_challenge_coverage,
)


def _request(index: int, challenges: list[str]) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-competitive",
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
        metadata={"benchmark_challenges": challenges},
    )
    request.refresh_hash()
    return request


def _complete_requests() -> list[NativeShotRequest]:
    return [
        _request(0, ["identity_consistency", "hands_anatomy"]),
        _request(1, ["multi_character_interaction", "object_interaction"]),
        _request(2, ["locomotion", "fast_camera_movement"]),
        _request(3, ["dialogue", "lighting_change"]),
        _request(4, ["physics", "identity_consistency"]),
    ]


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
    payload = f"challenge-video-{request.shot_id}".encode()
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


def test_complete_challenge_plan_covers_every_required_case():
    coverage = validate_challenge_coverage(_complete_requests())

    assert coverage.complete is True
    assert coverage.missing == ()
    assert set(coverage.challenge_to_shots) == set(REQUIRED_CHALLENGES)
    assert coverage.challenge_to_shots["dialogue"] == ("scene-competitive/shot-3",)


def test_challenge_plan_rejects_missing_difficult_case_before_gpu_execution(tmp_path):
    requests = _complete_requests()
    requests[4].metadata["benchmark_challenges"] = ["identity_consistency"]
    requests[4].refresh_hash()
    calls = []

    def executor(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(
            "GPU executor must not run for incomplete challenge coverage"
        )

    with pytest.raises(SeedanceStyleChallengeError, match="physics"):
        run_seedance_style_gpu_benchmark(
            "missing-physics",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=executor,
        )

    assert calls == []


def test_challenge_plan_rejects_unknown_self_declared_case():
    requests = _complete_requests()
    requests[0].metadata["benchmark_challenges"].append("easy_closeup")
    requests[0].refresh_hash()

    with pytest.raises(
        SeedanceStyleChallengeError, match="unsupported benchmark challenge"
    ):
        validate_challenge_coverage(requests)


def test_successful_challenge_run_binds_coverage_contract_to_gpu_manifest(tmp_path):
    requests = _complete_requests()

    def executor(request, profile, *, output_dir):
        assert profile is WAN22_TI2V_5B_PROFILE
        return _receipt(request, Path(output_dir))

    receipt = run_seedance_style_gpu_benchmark(
        "competitive-connected",
        requests,
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        shot_executor=executor,
    )

    payload = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
    contract = payload["competitive_challenge_contract"]
    assert contract["schema"] == "cineos-seedance-style-challenge-coverage/0.1"
    assert contract["complete"] is True
    assert contract["missing"] == []
    assert len(contract["contract_sha256"]) == 64
    assert set(contract["required_challenges"]) == set(REQUIRED_CHALLENGES)
    assert contract["challenge_to_shots"]["fast_camera_movement"] == [
        "scene-competitive/shot-2"
    ]


def test_challenge_metadata_change_must_be_rehashed_before_execution(tmp_path):
    requests = _complete_requests()
    requests[0].metadata["benchmark_challenges"].append("physics")

    def executor(*args, **kwargs):
        raise AssertionError("stale native request must fail before rendering")

    with pytest.raises(GPUConnectedBenchmarkError, match="missing or stale"):
        run_seedance_style_gpu_benchmark(
            "stale-challenge-metadata",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=executor,
        )
