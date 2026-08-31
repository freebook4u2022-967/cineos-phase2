"""Production-only quality-retry benchmark wrapper.

This module turns the generic connected quality-retry benchmark into an explicit
production-evidence gate. It never changes pretrained foundation provenance and
never upgrades injected/test execution into CINEOS-native capability.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .foundation_profiles import FoundationExecutionProfile
from .gpu_connected_benchmark import (
    GPUConnectedBenchmarkReceipt,
    _remove_stale_manifest,
)
from .gpu_foundation_smoke import execute_foundation_gpu_shot
from .gpu_quality_retry_benchmark import (
    GPUQualityRetryBenchmarkError,
    QualityEvaluator,
    ShotExecutor,
    run_quality_retry_connected_gpu_benchmark,
)
from .native_request import NativeShotRequest
from .quality_retry import QualityRetryPolicy
from .sequence_quality import ArtifactMeasuredSequenceQualityEvaluator


class ProductionGPUQualityRetryError(GPUQualityRetryBenchmarkError):
    """Raised when a quality-retry run lacks production GPU or measurement evidence."""


_INJECTED_RUNTIME_KWARGS = frozenset(
    {
        "torch_module",
        "reference_loader",
        "pipeline_factory",
        "video_exporter",
    }
)


def _validate_production_executor_kwargs(
    shot_executor_kwargs: dict[str, Any] | None,
) -> None:
    """Reject injected runtime boundaries before any production render begins.

    ``execute_foundation_gpu_shot`` deliberately exposes dependency-injection hooks
    for deterministic tests. A caller can therefore pass the real function object
    while still supplying a fake torch runtime, reference loader, Diffusers
    pipeline, or exporter through ``shot_executor_kwargs``. Runtime provenance
    would eventually downgrade that receipt, but a production milestone runner
    should fail *before* expensive rendering and before any temporary benchmark
    artifacts are created.

    Resource-selection options such as ``estimated_model_vram_gb`` and
    ``prefer_bfloat16`` remain legal because they tune the real runtime rather than
    substitute it.
    """

    if not shot_executor_kwargs:
        return
    injected = sorted(_INJECTED_RUNTIME_KWARGS.intersection(shot_executor_kwargs))
    if injected:
        raise ProductionGPUQualityRetryError(
            "production GPU benchmark forbids injected runtime boundary kwargs: "
            + ", ".join(injected)
        )


def run_production_quality_retry_connected_gpu_benchmark(
    benchmark_id: str,
    requests: Sequence[NativeShotRequest],
    profile: FoundationExecutionProfile,
    *,
    output_dir: str | Path,
    quality_evaluator: QualityEvaluator,
    retry_policy: QualityRetryPolicy | None = None,
    shot_executor: ShotExecutor = execute_foundation_gpu_shot,
    shot_executor_kwargs: dict[str, Any] | None = None,
) -> GPUConnectedBenchmarkReceipt:
    """Run a 5-10 shot benchmark requiring real GPU and artifact-bound QC evidence.

    The underlying benchmark remains reusable for deterministic regression tests.
    This production wrapper adds milestone rules: execution must use the actual
    default CINEOS CUDA + Diffusers executor, and QC must use an artifact-measured
    evaluator whose reports are cryptographically bound to rendered outputs.
    Injected executors, synthetic quality lambdas, CPU fallbacks, legacy receipts,
    stale metric reports, altered runtime provenance, or missing production QC
    evidence fail closed.
    """

    if shot_executor is not execute_foundation_gpu_shot:
        raise ProductionGPUQualityRetryError(
            "production GPU benchmark requires the unmodified default shot executor"
        )
    _validate_production_executor_kwargs(shot_executor_kwargs)
    if not isinstance(quality_evaluator, ArtifactMeasuredSequenceQualityEvaluator):
        raise ProductionGPUQualityRetryError(
            "production GPU benchmark requires artifact-measured sequence quality evidence"
        )

    receipt = run_quality_retry_connected_gpu_benchmark(
        benchmark_id,
        requests,
        profile,
        output_dir=output_dir,
        quality_evaluator=quality_evaluator,
        retry_policy=retry_policy,
        shot_executor=shot_executor,
        shot_executor_kwargs=shot_executor_kwargs,
    )
    manifest = Path(receipt.manifest_path)
    if not receipt.production_gpu_evidence:
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError(
            "production GPU evidence required, but the quality-retry benchmark did not "
            "run entirely through the unmodified default CUDA runtime"
        )
    if not receipt.production_quality_evidence:
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError(
            "production quality evidence required, but one or more accepted shots lack "
            "artifact-bound measured QC evidence"
        )
    if receipt.evidence_tier != "production-gpu-quality-gated":
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError(
            "production benchmark did not reach the production-gpu-quality-gated evidence tier"
        )
    return receipt


__all__ = [
    "ProductionGPUQualityRetryError",
    "run_production_quality_retry_connected_gpu_benchmark",
]
