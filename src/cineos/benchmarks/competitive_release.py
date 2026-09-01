"""Fail-closed release gate for the Seedance-class competitive benchmark.

This module deliberately validates *production evidence*, not architecture readiness.
A competitive claim requires the exact versioned suite, every mandatory case exactly
once, real GPU execution metadata, explicit external-foundation provenance, and
artifact-level validation for every case. Missing GPU evidence remains a blocker
rather than being converted into a synthetic pass.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from .exceptions import BenchmarkError
from .real_evidence import validate_real_inference_evidence
from .report import BenchmarkReport
from .seedance_competitive import seedance_competitive_suite


def validate_seedance_competitive_release(
    report: BenchmarkReport,
    *,
    case_output_dirs: Mapping[str, str | Path],
    foundation: Mapping[str, object],
) -> None:
    """Validate that a report is admissible as competitive production evidence.

    The gate is intentionally strict. It binds the report to the exact competitive
    suite content hash and renderer profile, requires one result per case, rejects
    warnings and duplicate/missing cases, verifies declared real-GPU metadata, and
    delegates each case to artifact-level real-inference validation.
    """

    suite = seedance_competitive_suite()
    if report.suite_id != suite.suite_id:
        raise BenchmarkError("competitive report suite_id does not match release contract")
    if report.suite_version != suite.suite_version:
        raise BenchmarkError("competitive report suite_version does not match release contract")
    if report.suite_hash != suite.content_hash:
        raise BenchmarkError("competitive report suite_hash does not match release contract")
    if report.renderer_profile != suite.renderer_profile:
        raise BenchmarkError("competitive report renderer_profile does not match release contract")
    if not report.hardware_profile.strip():
        raise BenchmarkError("competitive report must declare a hardware_profile")

    metadata = report.metadata
    if metadata.get("production_gpu_evidence") is not True:
        raise BenchmarkError("competitive report is not attested as production GPU evidence")
    if metadata.get("real_inference") is not True:
        raise BenchmarkError("competitive report is not attested as real inference")
    commit_sha = metadata.get("commit_sha")
    if not isinstance(commit_sha, str) or len(commit_sha.strip()) < 7:
        raise BenchmarkError("competitive report must bind evidence to a commit_sha")

    expected = {case.case_id: case for case in suite.cases if case.mandatory}
    seen: dict[str, object] = {}
    for result in report.results:
        if result.case_id in seen:
            raise BenchmarkError(f"duplicate competitive case result: {result.case_id}")
        seen[result.case_id] = result

    missing = sorted(set(expected) - set(seen))
    extra = sorted(set(seen) - set(expected))
    if missing:
        raise BenchmarkError("missing mandatory competitive case(s): " + ", ".join(missing))
    if extra:
        raise BenchmarkError("unexpected competitive case result(s): " + ", ".join(extra))
    if not report.passed:
        raise BenchmarkError("competitive benchmark report did not pass")

    for case_id, case in expected.items():
        result = seen[case_id]
        if result.warnings:
            raise BenchmarkError(f"competitive case contains warning(s): {case_id}")
        output_dir = case_output_dirs.get(case_id)
        if output_dir is None:
            raise BenchmarkError(f"missing output directory for competitive case: {case_id}")
        validate_real_inference_evidence(
            case,
            result,
            output_dir,
            foundation=foundation,
        )


__all__ = ["validate_seedance_competitive_release"]
