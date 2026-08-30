import json
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
        scene_id="scene-connected",
        camera={"movement": "tracking"},
        characters=[{"character_id": "lead"}],
        environment={"location": "street"},
        wardrobe=[],
        props=[],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={"action": "walk"},
        approved_reference_ids=["lead-approved-reference"],
        deterministic_seed=1000 + index,
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
    result = DiffusersVideoResult(
        shot_id=request.shot_id,
        scene_id=request.scene_id,
        output_path=str(artifact),
        frame_count=48,
        seed=request.deterministic_seed,
        foundation=WAN22_TI2V_5B_PROFILE.provenance,
        request_hash=request.content_hash,
    )
    import hashlib

    return GPUFoundationExecutionReceipt(
        result=result,
        execution_plan=_plan(),
        profile_id=WAN22_TI2V_5B_PROFILE.profile_id,
        origin=WAN22_TI2V_5B_PROFILE.origin,
        output_bytes=len(payload),
        output_sha256=hashlib.sha256(payload).hexdigest(),
        elapsed_seconds=0.25,
    )


def test_connected_gpu_benchmark_writes_manifest_only_after_five_fresh_shots(tmp_path):
    requests = [_request(index) for index in range(5)]

    def executor(request, profile, *, output_dir):
        assert profile is WAN22_TI2V_5B_PROFILE
        return _receipt(request, Path(output_dir))

    receipt = run_connected_gpu_benchmark(
        "seedance-style-smoke",
        requests,
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        shot_executor=executor,
    )

    assert len(receipt.shot_receipts) == 5
    assert receipt.total_output_bytes > 0
    assert len(receipt.chain_sha256) == 64
    manifest = Path(receipt.manifest_path)
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema"] == "cineos-gpu-connected-benchmark/0.2"
    assert payload["shot_count"] == 5
    assert payload["origin"] == "external_pretrained_foundation"
    assert (
        payload["foundation_profile"]["profile_id"] == WAN22_TI2V_5B_PROFILE.profile_id
    )
    assert [shot["request_hash"] for shot in payload["shots"]] == [
        request.content_hash for request in requests
    ]


def test_connected_gpu_benchmark_accepts_legacy_previous_shot_id_alias(tmp_path):
    requests = [_request(index) for index in range(5)]
    for request in requests:
        previous = request.continuity.pop("previous_shot")
        request.continuity["previous_shot_id"] = previous
        request.refresh_hash()

    def executor(request, profile, *, output_dir):
        assert profile is WAN22_TI2V_5B_PROFILE
        return _receipt(request, Path(output_dir))

    receipt = run_connected_gpu_benchmark(
        "legacy-continuity-alias",
        requests,
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        shot_executor=executor,
    )

    assert len(receipt.shot_receipts) == 5


def test_connected_gpu_benchmark_rejects_conflicting_continuity_aliases(tmp_path):
    requests = [_request(index) for index in range(5)]
    requests[2].continuity["previous_shot_id"] = "shot-0"
    requests[2].refresh_hash()
    calls = []

    def executor(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("executor must not run")

    with pytest.raises(GPUConnectedBenchmarkError, match="conflicting previous_shot"):
        run_connected_gpu_benchmark(
            "conflicting-aliases",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=executor,
        )

    assert calls == []


def test_connected_gpu_benchmark_rejects_stale_request_hash_before_execution(tmp_path):
    requests = [_request(index) for index in range(5)]
    requests[2].camera["movement"] = "changed-after-hash"
    calls = []

    def executor(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("executor must not run")

    with pytest.raises(GPUConnectedBenchmarkError, match="missing or stale"):
        run_connected_gpu_benchmark(
            "stale-request",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=executor,
        )

    assert calls == []


def test_connected_gpu_benchmark_rejects_broken_previous_shot_chain(tmp_path):
    requests = [_request(index) for index in range(5)]
    requests[3].continuity["previous_shot"] = "shot-0"
    requests[3].refresh_hash()
    calls = []

    def executor(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("executor must not run")

    with pytest.raises(GPUConnectedBenchmarkError, match="continuity chain is broken"):
        run_connected_gpu_benchmark(
            "broken-chain",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=executor,
        )

    assert calls == []


def test_connected_gpu_benchmark_rejects_previous_shot_on_first_clip(tmp_path):
    requests = [_request(index) for index in range(5)]
    requests[0].continuity["previous_shot"] = "shot-before-benchmark"
    requests[0].refresh_hash()

    with pytest.raises(
        GPUConnectedBenchmarkError, match="must not declare previous_shot"
    ):
        run_connected_gpu_benchmark(
            "bad-first-shot",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=lambda *args, **kwargs: None,
        )


def test_connected_gpu_benchmark_rejects_non_string_previous_shot(tmp_path):
    requests = [_request(index) for index in range(5)]
    requests[1].continuity["previous_shot"] = 123
    requests[1].refresh_hash()

    with pytest.raises(GPUConnectedBenchmarkError, match="must be a string or null"):
        run_connected_gpu_benchmark(
            "bad-link-type",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=lambda *args, **kwargs: None,
        )


def test_connected_gpu_benchmark_removes_stale_manifest_when_shot_fails(tmp_path):
    requests = [_request(index) for index in range(5)]
    manifest = tmp_path / "partial.gpu-benchmark.json"
    manifest.write_text('{"stale": true}\n', encoding="utf-8")
    calls = 0

    def executor(request, profile, *, output_dir):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("gpu render failed")
        return _receipt(request, Path(output_dir))

    with pytest.raises(RuntimeError, match="gpu render failed"):
        run_connected_gpu_benchmark(
            "partial",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=executor,
        )

    assert calls == 3
    assert not manifest.exists()


def test_connected_gpu_benchmark_enforces_seedance_style_shot_count(tmp_path):
    with pytest.raises(GPUConnectedBenchmarkError, match="between 5 and 10"):
        run_connected_gpu_benchmark(
            "too-short",
            [_request(index) for index in range(4)],
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=lambda *args, **kwargs: None,
        )
