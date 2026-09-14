"""Regression coverage for visual-conditioning lineage in serialized GPU evidence."""

from cineos.atlas.diffusers_video import DiffusersVideoResult, FoundationProvenance
from cineos.atlas.gpu_foundation_smoke import GPUFoundationExecutionReceipt
from cineos.atlas.gpu_preflight import GPUDeviceProfile, plan_gpu_execution
from cineos.atlas.production_diffusers import ProductionDiffusersVideoResult


def _plan():
    return plan_gpu_execution(
        GPUDeviceProfile(
            index=0,
            name="test-gpu",
            compute_capability=(9, 0),
            total_vram_gb=96.0,
            free_vram_gb=90.0,
            supports_bfloat16=True,
        ),
        estimated_model_vram_gb=80.0,
    )


def _receipt(result):
    return GPUFoundationExecutionReceipt(
        result=result,
        execution_plan=_plan(),
        profile_id="wan2.2-i2v-a14b-test",
        origin="external_pretrained_foundation",
        output_bytes=1024,
        output_sha256="a" * 64,
        elapsed_seconds=1.0,
        media_payload_bytes=900,
        runtime_provenance={
            "schema": "cineos-gpu-runtime-provenance/0.1",
            "runtime_mode": "default",
            "production_default_runtime": True,
            "cuda_device": "cuda:0",
            "dtype": "bfloat16",
            "injected_boundaries": {},
        },
    )


def _foundation():
    return FoundationProvenance(
        model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        revision="1" * 40,
        license_id="Apache-2.0",
        source_url="https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    )


def test_serialized_gpu_receipt_preserves_multi_reference_conditioning_lineage():
    result = ProductionDiffusersVideoResult(
        shot_id="shot-2",
        scene_id="scene-1",
        output_path="scene-1-shot-2.mp4",
        frame_count=81,
        seed=42,
        foundation=_foundation(),
        request_hash="b" * 64,
        artifact_sha256="c" * 64,
        artifact_size_bytes=1024,
        conditioning_provenance={
            "mode": "multi_reference_adapter",
            "consumed_reference_ids": ["hero-front", "partner-front"],
            "adapter_id": "cineos.production.reference_board",
            "adapter_version": "0.1.1",
        },
    )

    payload = _receipt(result).to_dict()

    assert payload["conditioning_provenance"] == result.conditioning_provenance
    assert payload["conditioning_provenance"]["consumed_reference_ids"] == [
        "hero-front",
        "partner-front",
    ]


def test_legacy_nonproduction_result_does_not_invent_conditioning_provenance():
    result = DiffusersVideoResult(
        shot_id="shot-1",
        scene_id="scene-1",
        output_path="scene-1-shot-1.mp4",
        frame_count=81,
        seed=42,
        foundation=_foundation(),
        request_hash="b" * 64,
    )

    payload = _receipt(result).to_dict()

    assert "conditioning_provenance" not in payload
