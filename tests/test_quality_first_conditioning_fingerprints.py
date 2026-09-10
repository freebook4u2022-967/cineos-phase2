"""Fail-closed conditioning fingerprint regressions for production GPU acceptance."""

from types import SimpleNamespace

import pytest

from cineos.atlas.gpu_benchmark_cli import GPUProductionBenchmarkCLIError
from cineos.atlas.quality_first_gpu_benchmark_cli import _validate_conditioning_binding


REF_A = "1" * 64
REF_B = "2" * 64
COMPOSED = "3" * 64


def _request(*reference_ids: str) -> SimpleNamespace:
    return SimpleNamespace(approved_reference_ids=list(reference_ids))


def _result(conditioning: object) -> SimpleNamespace:
    return SimpleNamespace(conditioning_provenance=conditioning)


def test_single_reference_requires_exact_consumed_content_fingerprint() -> None:
    _validate_conditioning_binding(
        _result(
            {
                "mode": "single_reference",
                "consumed_reference_ids": ["hero"],
                "consumed_reference_sha256": [REF_A],
                "conditioning_image_sha256": REF_A,
            }
        ),
        _request("hero"),
        shot_index=0,
    )


def test_single_reference_rejects_conditioning_image_substitution() -> None:
    with pytest.raises(GPUProductionBenchmarkCLIError, match="does not match consumed"):
        _validate_conditioning_binding(
            _result(
                {
                    "mode": "single_reference",
                    "consumed_reference_ids": ["hero"],
                    "consumed_reference_sha256": [REF_A],
                    "conditioning_image_sha256": REF_B,
                }
            ),
            _request("hero"),
            shot_index=1,
        )


def test_multi_reference_rejects_missing_or_malformed_consumed_hashes() -> None:
    with pytest.raises(GPUProductionBenchmarkCLIError, match="fingerprints"):
        _validate_conditioning_binding(
            _result(
                {
                    "mode": "multi_reference_adapter",
                    "consumed_reference_ids": ["hero", "partner"],
                    "consumed_reference_sha256": [REF_A, "not-a-sha256"],
                    "conditioning_image_sha256": COMPOSED,
                    "adapter_id": "cineos.reference-board",
                    "adapter_version": "1",
                }
            ),
            _request("hero", "partner"),
            shot_index=2,
        )


def test_multi_reference_rejects_duplicate_consumed_content() -> None:
    with pytest.raises(GPUProductionBenchmarkCLIError, match="duplicate consumed content"):
        _validate_conditioning_binding(
            _result(
                {
                    "mode": "multi_reference_adapter",
                    "consumed_reference_ids": ["hero", "partner"],
                    "consumed_reference_sha256": [REF_A, REF_A],
                    "conditioning_image_sha256": COMPOSED,
                    "adapter_id": "cineos.reference-board",
                    "adapter_version": "1",
                }
            ),
            _request("hero", "partner"),
            shot_index=3,
        )


def test_multi_reference_rejects_adapter_returning_one_source_unchanged() -> None:
    with pytest.raises(GPUProductionBenchmarkCLIError, match="unchanged source reference"):
        _validate_conditioning_binding(
            _result(
                {
                    "mode": "multi_reference_adapter",
                    "consumed_reference_ids": ["hero", "partner"],
                    "consumed_reference_sha256": [REF_A, REF_B],
                    "conditioning_image_sha256": REF_A,
                    "adapter_id": "cineos.reference-board",
                    "adapter_version": "1",
                }
            ),
            _request("hero", "partner"),
            shot_index=4,
        )


def test_multi_reference_accepts_distinct_consumed_and_composed_content() -> None:
    _validate_conditioning_binding(
        _result(
            {
                "mode": "multi_reference_adapter",
                "consumed_reference_ids": ["hero", "partner"],
                "consumed_reference_sha256": [REF_A, REF_B],
                "conditioning_image_sha256": COMPOSED,
                "adapter_id": "cineos.reference-board",
                "adapter_version": "1",
            }
        ),
        _request("hero", "partner"),
        shot_index=5,
    )


def test_legacy_result_without_conditioning_field_remains_compatible() -> None:
    _validate_conditioning_binding(
        SimpleNamespace(),
        _request("hero"),
        shot_index=6,
    )
