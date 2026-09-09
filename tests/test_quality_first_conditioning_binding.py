"""Regression coverage for production conditioning evidence binding."""

import pytest

from cineos.atlas.diffusers_video import FoundationProvenance
from cineos.atlas.gpu_benchmark_cli import GPUProductionBenchmarkCLIError
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_diffusers import ProductionDiffusersVideoResult
from cineos.atlas.quality_first_gpu_benchmark_cli import _validate_conditioning_binding


def _request(refs):
    request = NativeShotRequest(
        shot_id="shot-1",
        scene_id="scene-1",
        camera={"resolution": [832, 480], "fps": 16.0, "duration": 2.0},
        characters=[],
        environment={},
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=list(refs),
        deterministic_seed=42,
        renderer_requirements={
            "supported_resolution": [832, 480],
            "supported_fps": 16.0,
            "maximum_duration": 2.0,
        },
        metadata={},
    )
    request.refresh_hash()
    return request


def _result(request, conditioning):
    return ProductionDiffusersVideoResult(
        shot_id=request.shot_id,
        scene_id=request.scene_id,
        output_path="shot-1.mp4",
        frame_count=33,
        seed=42,
        foundation=FoundationProvenance(
            model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            revision="1" * 40,
            license_id="Apache-2.0",
            source_url="https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        ),
        request_hash=request.content_hash,
        artifact_sha256="a" * 64,
        artifact_size_bytes=1024,
        conditioning_provenance=conditioning,
    )


def test_quality_first_accepts_exact_multi_reference_conditioning_lineage():
    request = _request(("hero-front", "partner-front"))
    result = _result(
        request,
        {
            "mode": "multi_reference_adapter",
            "consumed_reference_ids": ["hero-front", "partner-front"],
            "adapter_id": "cineos.production.reference_board",
            "adapter_version": "0.1.1",
        },
    )

    _validate_conditioning_binding(result, request, shot_index=0)


def test_quality_first_rejects_partial_reference_consumption():
    request = _request(("hero-front", "partner-front"))
    result = _result(
        request,
        {
            "mode": "multi_reference_adapter",
            "consumed_reference_ids": ["hero-front"],
            "adapter_id": "cineos.production.reference_board",
            "adapter_version": "0.1.1",
        },
    )

    with pytest.raises(GPUProductionBenchmarkCLIError, match="approved reference board"):
        _validate_conditioning_binding(result, request, shot_index=2)


def test_quality_first_rejects_reordered_reference_consumption():
    request = _request(("hero-front", "partner-front"))
    result = _result(
        request,
        {
            "mode": "multi_reference_adapter",
            "consumed_reference_ids": ["partner-front", "hero-front"],
            "adapter_id": "cineos.production.reference_board",
            "adapter_version": "0.1.1",
        },
    )

    with pytest.raises(GPUProductionBenchmarkCLIError, match="approved reference board"):
        _validate_conditioning_binding(result, request, shot_index=1)


def test_quality_first_rejects_missing_multi_reference_adapter_provenance():
    request = _request(("hero-front", "partner-front"))
    result = _result(
        request,
        {
            "mode": "multi_reference_adapter",
            "consumed_reference_ids": ["hero-front", "partner-front"],
            "adapter_id": "",
            "adapter_version": "0.1.1",
        },
    )

    with pytest.raises(GPUProductionBenchmarkCLIError, match="adapter provenance"):
        _validate_conditioning_binding(result, request, shot_index=4)


def test_quality_first_rejects_conditioning_when_request_has_no_references():
    request = _request(())
    result = _result(
        request,
        {
            "mode": "single_reference",
            "consumed_reference_ids": ["unapproved"],
        },
    )

    with pytest.raises(GPUProductionBenchmarkCLIError, match="no approved references"):
        _validate_conditioning_binding(result, request, shot_index=0)
