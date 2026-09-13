"""Regression tests for immutable approved-reference conditioning evidence."""

from types import SimpleNamespace

import pytest

from cineos.atlas.gpu_benchmark_cli import GPUProductionBenchmarkCLIError
from cineos.atlas.quality_first_gpu_benchmark_cli import _validate_conditioning_binding


def _request(*reference_ids: str):
    return SimpleNamespace(approved_reference_ids=list(reference_ids))


def _multi_reference_result(*consumed_hashes: str):
    return SimpleNamespace(
        conditioning_provenance={
            "mode": "multi_reference_adapter",
            "consumed_reference_ids": ["hero", "partner"],
            "consumed_reference_sha256": list(consumed_hashes),
            "conditioning_image_sha256": "3" * 64,
            "adapter_id": "cineos-native-reference-compositor",
            "adapter_version": "1",
        }
    )


def test_conditioning_binding_accepts_exact_approved_reference_bytes():
    hero_hash = "1" * 64
    partner_hash = "2" * 64

    _validate_conditioning_binding(
        _multi_reference_result(hero_hash, partner_hash),
        _request("hero", "partner"),
        shot_index=0,
        expected_reference_hashes=(hero_hash, partner_hash),
    )


def test_conditioning_binding_rejects_substituted_content_under_approved_id():
    hero_hash = "1" * 64
    partner_hash = "2" * 64
    substituted_partner_hash = "4" * 64

    with pytest.raises(
        GPUProductionBenchmarkCLIError,
        match="consumed reference content does not match approved manifest bytes",
    ):
        _validate_conditioning_binding(
            _multi_reference_result(hero_hash, substituted_partner_hash),
            _request("hero", "partner"),
            shot_index=2,
            expected_reference_hashes=(hero_hash, partner_hash),
        )


def test_conditioning_binding_rejects_reordered_reference_bytes():
    hero_hash = "1" * 64
    partner_hash = "2" * 64

    with pytest.raises(
        GPUProductionBenchmarkCLIError,
        match="consumed reference content does not match approved manifest bytes",
    ):
        _validate_conditioning_binding(
            _multi_reference_result(partner_hash, hero_hash),
            _request("hero", "partner"),
            shot_index=1,
            expected_reference_hashes=(hero_hash, partner_hash),
        )


def test_conditioning_binding_rejects_invalid_approved_digest_contract():
    with pytest.raises(
        GPUProductionBenchmarkCLIError,
        match="approved reference fingerprints are missing or invalid",
    ):
        _validate_conditioning_binding(
            _multi_reference_result("1" * 64, "2" * 64),
            _request("hero", "partner"),
            shot_index=3,
            expected_reference_hashes=("not-a-sha256", "2" * 64),
        )
