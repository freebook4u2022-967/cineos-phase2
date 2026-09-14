"""Fail-closed release gate for the Seedance-class competitive benchmark.

This module deliberately validates *production evidence*, not architecture readiness.
A competitive claim requires the exact versioned suite, every mandatory case exactly
once, real GPU execution metadata, explicit external-foundation provenance, an
immutable upstream foundation revision, and content-addressed artifact validation for
every case. Missing GPU evidence remains a blocker rather than being converted into a
synthetic pass.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from .exceptions import BenchmarkError
from .real_evidence import validate_real_inference_evidence
from .report import BenchmarkReport
from .seedance_competitive import seedance_competitive_suite

_FULL_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FOUNDATION_REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")


def _validated_commit_sha(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not _FULL_GIT_SHA_RE.fullmatch(value):
        raise BenchmarkError(
            f"{field_name} must be a full lowercase 40-hex Git commit SHA"
        )
    return value


def _validated_foundation_revision(value: object) -> str:
    """Require an immutable content revision for an external foundation.

    Hugging Face and compatible model registries commonly expose immutable commit
    identifiers as hexadecimal object IDs. Branches such as ``main`` and mutable tags
    are intentionally rejected here: a production benchmark must remain reproducible
    even if the upstream repository later changes.
    """

    if not isinstance(value, str) or not _FOUNDATION_REVISION_RE.fullmatch(value):
        raise BenchmarkError(
            "competitive foundation revision must be an immutable lowercase "
            "40-64 hex object ID"
        )
    return value


def validate_seedance_competitive_release(
    report: BenchmarkReport,
    *,
    case_output_dirs: Mapping[str, str | Path],
    foundation: Mapping[str, object],
    expected_commit_sha: str | None = None,
    expected_foundation_revision: str | None = None,
) -> None:
    """Validate that a report is admissible as competitive production evidence.

    The gate is intentionally strict. It binds the report to the exact competitive
    suite content hash and renderer profile, requires one result per case, rejects
    warnings and duplicate/missing cases, verifies declared real-GPU metadata, pins
    the external foundation to an immutable upstream revision, and delegates each
    case to content-addressed real-inference validation.

    When ``expected_commit_sha`` is supplied by a release workflow, the evidence must
    name that exact checkout. This prevents a valid artifact bundle from being
    relabelled as evidence for another CINEOS revision. Likewise,
    ``expected_foundation_revision`` can bind release validation to the exact external
    pretrained weights resolved by the GPU workflow.
    """

    suite = seedance_competitive_suite()
    if report.suite_id != suite.suite_id:
        raise BenchmarkError(
            "competitive report suite_id does not match release contract"
        )
    if report.suite_version != suite.suite_version:
        raise BenchmarkError(
            "competitive report suite_version does not match release contract"
        )
    if report.suite_hash != suite.content_hash:
        raise BenchmarkError(
            "competitive report suite_hash does not match release contract"
        )
    if report.renderer_profile != suite.renderer_profile:
        raise BenchmarkError(
            "competitive report renderer_profile does not match release contract"
        )
    if not report.hardware_profile.strip():
        raise BenchmarkError("competitive report must declare a hardware_profile")

    metadata = report.metadata
    if metadata.get("production_gpu_evidence") is not True:
        raise BenchmarkError(
            "competitive report is not attested as production GPU evidence"
        )
    if metadata.get("real_inference") is not True:
        raise BenchmarkError("competitive report is not attested as real inference")

    commit_sha = _validated_commit_sha(
        metadata.get("commit_sha"), field_name="competitive report commit_sha"
    )
    if expected_commit_sha is not None:
        expected = _validated_commit_sha(
            expected_commit_sha, field_name="expected_commit_sha"
        )
        if commit_sha != expected:
            raise BenchmarkError(
                "competitive report commit_sha does not match release checkout"
            )

    if foundation.get("origin") != "external_pretrained_foundation":
        raise BenchmarkError(
            "competitive foundation must declare external_pretrained_foundation origin"
        )
    model_id = foundation.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise BenchmarkError("competitive foundation must declare model_id")
    foundation_revision = _validated_foundation_revision(foundation.get("revision"))
    if expected_foundation_revision is not None:
        expected_revision = _validated_foundation_revision(expected_foundation_revision)
        if foundation_revision != expected_revision:
            raise BenchmarkError(
                "competitive foundation revision does not match release foundation"
            )

    expected = {case.case_id: case for case in suite.cases if case.mandatory}
    seen: dict[str, object] = {}
    for result in report.results:
        if result.case_id in seen:
            raise BenchmarkError(f"duplicate competitive case result: {result.case_id}")
        seen[result.case_id] = result

    missing = sorted(set(expected) - set(seen))
    extra = sorted(set(seen) - set(expected))
    if missing:
        raise BenchmarkError(
            "missing mandatory competitive case(s): " + ", ".join(missing)
        )
    if extra:
        raise BenchmarkError(
            "unexpected competitive case result(s): " + ", ".join(extra)
        )
    if not report.passed:
        raise BenchmarkError("competitive benchmark report did not pass")

    for case_id, case in expected.items():
        result = seen[case_id]
        if result.warnings:
            raise BenchmarkError(f"competitive case contains warning(s): {case_id}")
        output_dir = case_output_dirs.get(case_id)
        if output_dir is None:
            raise BenchmarkError(
                f"missing output directory for competitive case: {case_id}"
            )
        validate_real_inference_evidence(
            case,
            result,
            output_dir,
            foundation=foundation,
            require_artifact_manifest=True,
        )


__all__ = ["validate_seedance_competitive_release"]
