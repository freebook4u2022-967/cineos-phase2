from cineos.atlas.diffusers_video import DiffusersVideoResult
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_connected_benchmark import GPUConnectedBenchmarkReceipt
from cineos.atlas.gpu_foundation_smoke import GPUFoundationExecutionReceipt
from cineos.atlas.gpu_preflight import GPUExecutionPlan


def _plan(device: str = "cuda:0") -> GPUExecutionPlan:
    return GPUExecutionPlan(
        device=device,
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


def _shot(index: int, *, runtime_provenance=None) -> GPUFoundationExecutionReceipt:
    result = DiffusersVideoResult(
        shot_id=f"shot-{index}",
        scene_id="scene-evidence",
        output_path=f"/tmp/scene-evidence-shot-{index}.mp4",
        frame_count=48,
        seed=1000 + index,
        foundation=WAN22_TI2V_5B_PROFILE.provenance,
        request_hash=f"request-{index}",
    )
    return GPUFoundationExecutionReceipt(
        result=result,
        execution_plan=_plan(),
        profile_id=WAN22_TI2V_5B_PROFILE.profile_id,
        origin=WAN22_TI2V_5B_PROFILE.origin,
        output_bytes=1024 + index,
        output_sha256=f"{index + 1:064x}",
        elapsed_seconds=1.0,
        media_payload_bytes=512,
        runtime_provenance=runtime_provenance,
    )


def _benchmark(*receipts, quality_reports=()):
    return GPUConnectedBenchmarkReceipt(
        benchmark_id="evidence-tier",
        profile_id=WAN22_TI2V_5B_PROFILE.profile_id,
        origin=WAN22_TI2V_5B_PROFILE.origin,
        shot_receipts=tuple(receipts),
        chain_sha256="a" * 64,
        total_output_bytes=sum(item.output_bytes for item in receipts),
        elapsed_seconds=5.0,
        manifest_path="/tmp/evidence-tier.gpu-benchmark.json",
        quality_reports=tuple(quality_reports),
    )


def _default_runtime(device: str = "cuda:0"):
    return {
        "schema": "cineos-gpu-runtime-provenance/0.1",
        "runtime_mode": "default",
        "production_default_runtime": True,
        "cuda_device": device,
        "dtype": "bfloat16",
        "injected_boundaries": {
            "torch_module": False,
            "reference_loader": False,
            "pipeline_factory": False,
            "video_exporter": False,
        },
    }


def _measured_report(index: int):
    digest = f"{index + 1:064x}"
    return {
        "accepted": True,
        "score": 0.9,
        "production_measurement_evidence": True,
        "output_sha256": digest,
        "scene_id": "scene-evidence",
        "shot_id": f"shot-{index}",
        "effective_request_hash": f"request-{index}",
        "measurement": {
            "schema": "cineos-sequence-quality-measurement/0.1",
            "observer_id": "cineos-artifact-video-observer/0.1",
            "artifact_sha256": digest,
        },
    }


def test_legacy_or_test_receipts_cannot_be_claimed_as_production_gpu_evidence():
    receipt = _benchmark(*[_shot(index) for index in range(5)])

    assert receipt.production_gpu_evidence is False
    assert receipt.evidence_tier == "non-production-or-injected"
    payload = receipt.to_dict()
    assert payload["production_gpu_evidence"] is False
    assert payload["evidence_tier"] == "non-production-or-injected"


def test_all_default_cuda_receipts_are_production_execution_evidence():
    receipt = _benchmark(
        *[_shot(index, runtime_provenance=_default_runtime()) for index in range(5)]
    )

    assert receipt.production_gpu_evidence is True
    assert receipt.evidence_tier == "production-gpu-execution"


def test_generic_quality_reports_cannot_claim_strongest_evidence_tier():
    receipt = _benchmark(
        *[_shot(index, runtime_provenance=_default_runtime()) for index in range(5)],
        quality_reports=({"accepted": True, "score": 0.9},) * 5,
    )

    assert receipt.production_gpu_evidence is True
    assert receipt.production_quality_evidence is False
    assert receipt.evidence_tier == "production-gpu-execution"


def test_exact_artifact_bound_quality_gets_strongest_evidence_tier():
    receipt = _benchmark(
        *[_shot(index, runtime_provenance=_default_runtime()) for index in range(5)],
        quality_reports=tuple(_measured_report(index) for index in range(5)),
    )

    assert receipt.production_gpu_evidence is True
    assert receipt.production_quality_evidence is True
    assert receipt.evidence_tier == "production-gpu-quality-gated"


def test_one_injected_receipt_downgrades_entire_connected_sequence():
    receipts = [
        _shot(index, runtime_provenance=_default_runtime()) for index in range(5)
    ]
    receipts[3] = _shot(
        3,
        runtime_provenance={
            **_default_runtime(),
            "runtime_mode": "injected",
            "production_default_runtime": False,
        },
    )

    receipt = _benchmark(*receipts)

    assert receipt.production_gpu_evidence is False
    assert receipt.evidence_tier == "non-production-or-injected"


def test_non_cuda_runtime_cannot_be_promoted_to_production_gpu_evidence():
    provenance = _default_runtime(device="cpu")
    receipt = _benchmark(
        *[_shot(index, runtime_provenance=provenance) for index in range(5)]
    )

    assert receipt.production_gpu_evidence is False
