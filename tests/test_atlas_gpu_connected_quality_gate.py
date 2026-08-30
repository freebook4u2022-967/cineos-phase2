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
        scene_id="scene-quality-gated",
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


def _executor(request, profile, *, output_dir):
    assert profile is WAN22_TI2V_5B_PROFILE
    return _receipt(request, Path(output_dir))


def _accepted_report(output_path, *, shot, attempt_index):
    assert Path(output_path).is_file()
    assert attempt_index == 0
    return {
        "schema": "cineos-sequence-quality-report/0.1",
        "accepted": True,
        "decision": "accept",
        "score": 0.91,
        "metrics": {
            "identity_similarity": 0.94,
            "temporal_consistency": 0.90,
            "artifact_integrity": 1.0,
            "motion_quality": 0.86,
        },
        "failed_metrics": [],
        "directives": [],
        "observer": f"measured-{shot.shot_id}",
    }


def test_connected_gpu_benchmark_persists_quality_evidence_for_every_shot(tmp_path):
    requests = [_request(index) for index in range(5)]

    receipt = run_connected_gpu_benchmark(
        "quality-gated-connected",
        requests,
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        shot_executor=_executor,
        quality_evaluator=_accepted_report,
    )

    assert len(receipt.quality_reports) == 5
    assert [report["shot_id"] for report in receipt.quality_reports] == [
        request.shot_id for request in requests
    ]
    assert [report["request_hash"] for report in receipt.quality_reports] == [
        request.content_hash for request in requests
    ]

    payload = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
    assert payload["quality_gate_applied"] is True
    assert len(payload["quality_reports"]) == 5
    assert all(report["accepted"] for report in payload["quality_reports"])


def test_connected_gpu_benchmark_rejects_failed_measured_quality_without_manifest(tmp_path):
    requests = [_request(index) for index in range(5)]
    manifest = tmp_path / "quality-rejected.gpu-benchmark.json"

    def evaluator(output_path, *, shot, attempt_index):
        report = _accepted_report(
            output_path,
            shot=shot,
            attempt_index=attempt_index,
        )
        if shot.shot_id == "shot-2":
            report["accepted"] = False
            report["decision"] = "reject"
            report["score"] = 0.62
            report["failed_metrics"] = [
                "identity_similarity",
                "temporal_consistency",
            ]
        return report

    with pytest.raises(
        GPUConnectedBenchmarkError,
        match="failed connected quality gate: identity_similarity, temporal_consistency",
    ):
        run_connected_gpu_benchmark(
            "quality-rejected",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=_executor,
            quality_evaluator=evaluator,
        )

    assert not manifest.exists()


def test_connected_gpu_benchmark_rejects_malformed_quality_report(tmp_path):
    requests = [_request(index) for index in range(5)]

    with pytest.raises(GPUConnectedBenchmarkError, match="missing boolean accepted"):
        run_connected_gpu_benchmark(
            "quality-malformed",
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=tmp_path,
            shot_executor=_executor,
            quality_evaluator=lambda *_args, **_kwargs: {"score": 0.9},
        )


def test_connected_gpu_benchmark_remains_backward_compatible_without_quality_gate(tmp_path):
    requests = [_request(index) for index in range(5)]

    receipt = run_connected_gpu_benchmark(
        "execution-only",
        requests,
        WAN22_TI2V_5B_PROFILE,
        output_dir=tmp_path,
        shot_executor=_executor,
    )

    payload = json.loads(Path(receipt.manifest_path).read_text(encoding="utf-8"))
    assert receipt.quality_reports == ()
    assert payload["quality_gate_applied"] is False
    assert payload["quality_reports"] == []
