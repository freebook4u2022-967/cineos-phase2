import hashlib
import json
from pathlib import Path

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoResult
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_connected_benchmark import (
    GPUConnectedBenchmarkError,
    GPUConnectedBenchmarkReceipt,
)
from cineos.atlas.gpu_foundation_smoke import GPUFoundationExecutionReceipt
from cineos.atlas.gpu_preflight import GPUExecutionPlan
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.seedance_style_challenge import (
    REQUIRED_CHALLENGES,
    ChallengeBoundGPUConnectedBenchmarkReceipt,
    SeedanceStyleChallengeError,
    run_seedance_style_gpu_benchmark,
    validate_challenge_coverage,
)


def _request(index: int, challenges: list[str]) -> NativeShotRequest:
    performance = {"action": "walk"}
    if "dialogue" in challenges:
        performance["dialogue_timing"] = [
            {
                "start_seconds": 0.25,
                "end_seconds": 1.25,
                "speaker_id": "lead",
                "text": "Stay with me.",
            }
        ]

    characters = [{"character_id": "lead"}]
    approved_reference_ids = ["lead-approved-reference"]
    props = []
    if "multi_character_interaction" in challenges:
        characters.append({"character_id": "support"})
        approved_reference_ids.append("support-approved-reference")
    if "object_interaction" in challenges:
        props.append({"prop_id": "parcel", "description": "sealed parcel"})

    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-competitive",
        camera={"movement": "tracking"},
        characters=characters,
        environment={"location": "street"},
        wardrobe=[],
        props=props,
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance=performance,
        approved_reference_ids=approved_reference_ids,
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


def test_multi_character_challenge_requires_two_conditioned_identities_before_gpu(tmp_path):
    requests = _complete_requests()
    requests[1].characters = [{"character_id": "lead"}]
    requests[1].approved_reference_ids = ["lead-approved-reference"]
    requests[1].refresh_hash()
    calls = []

    def executor(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("structurally invalid challenge must fail before GPU")

    with pytest.raises(
        SeedanceStyleChallengeError, match="fewer than two distinct conditioned"
    ):
        run_seedance_style_gpu_benchmark(
            "single-character-interaction",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=executor,
        )

    assert calls == []


def test_object_interaction_challenge_requires_declared_prop_before_gpu(tmp_path):
    requests = _complete_requests()
    requests[1].props = []
    requests[1].refresh_hash()
    calls = []

    def executor(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("structurally invalid challenge must fail before GPU")

    with pytest.raises(SeedanceStyleChallengeError, match="has no declared props"):
        run_seedance_style_gpu_benchmark(
            "object-interaction-without-prop",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=executor,
        )

    assert calls == []


def test_successful_challenge_run_binds_coverage_contract_to_receipt_and_manifest(
    tmp_path,
):
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

    assert isinstance(receipt, GPUConnectedBenchmarkReceipt)
    assert isinstance(receipt, ChallengeBoundGPUConnectedBenchmarkReceipt)
    serialized_contract = receipt.to_dict()["competitive_challenge_contract"]
    assert serialized_contract["complete"] is True
    assert serialized_contract["missing"] == []

    payload = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
    contract = payload["competitive_challenge_contract"]
    assert contract == serialized_contract
    assert contract["schema"] == "cineos-seedance-style-challenge-coverage/0.2"
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
