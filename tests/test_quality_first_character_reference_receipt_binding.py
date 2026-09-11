"""Regression tests for downstream character-to-reference receipt ownership."""

from types import SimpleNamespace

import pytest

from cineos.atlas.gpu_benchmark_cli import GPUProductionBenchmarkCLIError
from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.quality_first_gpu_benchmark_cli import _validate_conditioning_binding


def _request(*, mapped: bool = True) -> NativeShotRequest:
    characters = [
        {"character_uuid": "hero"},
        {"character_uuid": "partner"},
    ]
    if mapped:
        characters[0]["approved_reference_ids"] = ["hero-ref"]
        characters[1]["approved_reference_ids"] = ["partner-ref"]
    request = NativeShotRequest(
        shot_id="shot-0",
        scene_id="identity-receipt-binding",
        camera={"resolution": [832, 480], "fps": 16.0, "duration": 2.0},
        characters=characters,
        environment={},
        wardrobe=[],
        props=[],
        continuity={"previous_shot": None},
        performance={},
        approved_reference_ids=["hero-ref", "partner-ref"],
        deterministic_seed=7,
        renderer_requirements={
            "supported_resolution": [832, 480],
            "supported_fps": 16.0,
            "maximum_duration": 2.0,
        },
    )
    request.refresh_hash()
    return request


def _result(bindings):
    return SimpleNamespace(
        conditioning_provenance={
            "mode": "multi_reference_adapter",
            "consumed_reference_ids": ["hero-ref", "partner-ref"],
            "consumed_character_reference_ids": bindings,
            "consumed_reference_sha256": ["1" * 64, "2" * 64],
            "conditioning_image_sha256": "3" * 64,
            "adapter_id": "cineos-test-identity-adapter",
            "adapter_version": "1",
        }
    )


def _expected_bindings():
    return [
        {"character_uuid": "hero", "reference_ids": ["hero-ref"]},
        {"character_uuid": "partner", "reference_ids": ["partner-ref"]},
    ]


def test_receipt_accepts_exact_character_reference_ownership():
    _validate_conditioning_binding(
        _result(_expected_bindings()),
        _request(),
        shot_index=0,
    )


def test_receipt_rejects_character_reference_swap_with_same_global_reference_set():
    swapped = [
        {"character_uuid": "hero", "reference_ids": ["partner-ref"]},
        {"character_uuid": "partner", "reference_ids": ["hero-ref"]},
    ]

    with pytest.raises(GPUProductionBenchmarkCLIError, match="ownership contract"):
        _validate_conditioning_binding(_result(swapped), _request(), shot_index=0)


def test_receipt_rejects_missing_character_reference_ownership_evidence():
    result = _result(_expected_bindings())
    del result.conditioning_provenance["consumed_character_reference_ids"]

    with pytest.raises(
        GPUProductionBenchmarkCLIError,
        match="missing character-to-reference",
    ):
        _validate_conditioning_binding(result, _request(), shot_index=0)


def test_receipt_rejects_extra_character_reference_ownership_binding():
    extra = _expected_bindings() + [
        {"character_uuid": "intruder", "reference_ids": ["hero-ref"]}
    ]

    with pytest.raises(GPUProductionBenchmarkCLIError, match="ownership contract"):
        _validate_conditioning_binding(_result(extra), _request(), shot_index=0)


def test_legacy_unmapped_request_does_not_require_character_ownership_evidence():
    result = _result(None)
    del result.conditioning_provenance["consumed_character_reference_ids"]

    _validate_conditioning_binding(result, _request(mapped=False), shot_index=0)
