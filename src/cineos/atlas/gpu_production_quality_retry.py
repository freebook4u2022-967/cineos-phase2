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
from .gpu_persistent_session import PersistentGPUFoundationExecutor
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
_RESOURCE_RUNTIME_KWARGS = frozenset(
    {
        "estimated_model_vram_gb",
        "prefer_bfloat16",
    }
)


def _validate_production_executor_kwargs(
    shot_executor_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate real-runtime resource options before any production render begins.

    ``execute_foundation_gpu_shot`` deliberately exposes dependency-injection hooks
    for deterministic tests. A caller can therefore pass the real function object
    while still supplying a fake torch runtime, reference loader, Diffusers
    pipeline, or exporter through ``shot_executor_kwargs``. A production milestone
    runner must reject those substitutions before expensive rendering begins.

    Only resource-selection options are accepted here. They are consumed by the
    persistent model session and never forwarded as per-shot runtime overrides.
    """

    if not shot_executor_kwargs:
        return {}
    supplied = dict(shot_executor_kwargs)
    injected = sorted(_INJECTED_RUNTIME_KWARGS.intersection(supplied))
    if injected:
        raise ProductionGPUQualityRetryError(
            "production GPU benchmark forbids injected runtime boundary kwargs: "
            + ", ".join(injected)
        )
    unsupported = sorted(set(supplied).difference(_RESOURCE_RUNTIME_KWARGS))
    if unsupported:
        raise ProductionGPUQualityRetryError(
            "production GPU benchmark received unsupported runtime kwargs: "
            + ", ".join(unsupported)
        )
    return supplied


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

    The production path keeps one selected pretrained foundation resident for the
    entire connected sequence, including quality-driven retries. This avoids paying
    model load and warmup for every attempt while preserving per-shot request hashes,
    fresh artifacts, runtime provenance, QC evidence, and retry lineage.

    The public ``shot_executor`` argument remains for backwards-compatible validation
    but production execution rejects replacements. The wrapper itself owns the
    persistent executor so an injected callable cannot be promoted to production
    evidence.
    """

    if shot_executor is not execute_foundation_gpu_shot:
        raise ProductionGPUQualityRetryError(
            "production GPU benchmark requires the unmodified default shot executor"
        )
    runtime_options = _validate_production_executor_kwargs(shot_executor_kwargs)
    if not isinstance(quality_evaluator, ArtifactMeasuredSequenceQualityEvaluator):
        raise ProductionGPUQualityRetryError(
            "production GPU benchmark requires artifact-measured sequence quality evidence"
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    persistent_kwargs: dict[str, Any] = {}
    if "estimated_model_vram_gb" in runtime_options:
        persistent_kwargs["estimated_model_vram_gb"] = runtime_options[
            "estimated_model_vram_gb"
        ]
    if "prefer_bfloat16" in runtime_options:
        persistent_kwargs["prefer_bfloat16"] = runtime_options["prefer_bfloat16"]

    with PersistentGPUFoundationExecutor(
        profile,
        output_dir=output_root,
        **persistent_kwargs,
    ) as persistent_executor:
        receipt = run_quality_retry_connected_gpu_benchmark(
            benchmark_id,
            requests,
            profile,
            output_dir=output_root,
            quality_evaluator=quality_evaluator,
            retry_policy=retry_policy,
            shot_executor=persistent_executor,
            shot_executor_kwargs=None,
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
