"""Production-only quality-retry benchmark wrappers.

These wrappers turn the generic connected quality-retry benchmark into explicit
production-evidence gates. They never change pretrained foundation provenance and
never upgrade injected/test execution into CINEOS-native capability.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .foundation_profiles import FoundationExecutionProfile
from .gpu_connected_benchmark import (
    GPUConnectedBenchmarkReceipt,
    _remove_stale_manifest,
    _validate_requests,
)
from .gpu_foundation_smoke import execute_foundation_gpu_shot
from .gpu_persistent_session import PersistentGPUFoundationExecutor
from .gpu_quality_retry_benchmark import (
    GPUQualityRetryBenchmarkError,
    QualityEvaluator,
    ShotExecutor,
    TransitionEvaluator,
    run_quality_retry_connected_gpu_benchmark,
)
from .native_request import NativeShotRequest
from .quality_retry import QualityRetryPolicy
from .sequence_quality import ArtifactMeasuredSequenceQualityEvaluator
from .transition_quality import ArtifactMeasuredTransitionQualityEvaluator


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
    """Validate real-runtime resource options before any production render begins."""

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
    transition_evaluator: TransitionEvaluator | None = None,
    retry_policy: QualityRetryPolicy | None = None,
    shot_executor: ShotExecutor = execute_foundation_gpu_shot,
    shot_executor_kwargs: dict[str, Any] | None = None,
) -> GPUConnectedBenchmarkReceipt:
    """Run 5-10 shots requiring real GPU and artifact-bound shot QC evidence.

    ``transition_evaluator`` is optional here for backwards compatibility. When it
    is supplied on the production path it must be the attested artifact-measured
    evaluator; arbitrary injected seam scorers cannot become production evidence.
    New competitive continuity validation should use the stricter dedicated entry
    point below, where transition evidence is mandatory.
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
    if transition_evaluator is not None and not isinstance(
        transition_evaluator,
        ArtifactMeasuredTransitionQualityEvaluator,
    ):
        raise ProductionGPUQualityRetryError(
            "production transition QC requires an artifact-measured transition evaluator"
        )

    _validate_requests(requests)
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
            transition_evaluator=transition_evaluator,
            retry_policy=retry_policy,
            shot_executor=persistent_executor,
            shot_executor_kwargs=None,
        )

    manifest = Path(receipt.manifest_path)
    if not receipt.production_gpu_evidence:
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError(
            "production GPU evidence required, but benchmark execution was not "
            "entirely through the unmodified CUDA runtime"
        )
    if not receipt.production_quality_evidence:
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError(
            "production quality evidence required for every accepted shot"
        )
    if receipt.evidence_tier != "production-gpu-quality-gated":
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError(
            "production benchmark did not reach the required quality-gated tier"
        )
    return receipt


def run_production_continuity_quality_retry_connected_gpu_benchmark(
    benchmark_id: str,
    requests: Sequence[NativeShotRequest],
    profile: FoundationExecutionProfile,
    *,
    output_dir: str | Path,
    quality_evaluator: ArtifactMeasuredSequenceQualityEvaluator,
    transition_evaluator: ArtifactMeasuredTransitionQualityEvaluator,
    retry_policy: QualityRetryPolicy | None = None,
    shot_executor: ShotExecutor = execute_foundation_gpu_shot,
    shot_executor_kwargs: dict[str, Any] | None = None,
) -> GPUConnectedBenchmarkReceipt:
    """Run the strict production sequence gate with mandatory cross-shot seam QC."""

    if not isinstance(
        transition_evaluator,
        ArtifactMeasuredTransitionQualityEvaluator,
    ):
        raise ProductionGPUQualityRetryError(
            "competitive production continuity requires attested transition evidence"
        )
    receipt = run_production_quality_retry_connected_gpu_benchmark(
        benchmark_id,
        requests,
        profile,
        output_dir=output_dir,
        quality_evaluator=quality_evaluator,
        transition_evaluator=transition_evaluator,
        retry_policy=retry_policy,
        shot_executor=shot_executor,
        shot_executor_kwargs=shot_executor_kwargs,
    )
    manifest = Path(receipt.manifest_path)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError(
            "cannot verify production continuity benchmark manifest"
        ) from exc
    gate = payload.get("quality_retry_gate")
    expected = len(requests) - 1
    if not isinstance(gate, dict):
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError("continuity quality gate is missing")
    if gate.get("transition_gate_applied") is not True:
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError(
            "production transition gate was not applied"
        )
    if gate.get("accepted_transition_count") != expected:
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError(
            "production transition evidence is incomplete for the connected sequence"
        )
    transitions = gate.get("accepted_transitions")
    if not isinstance(transitions, list) or len(transitions) != expected:
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError(
            "production transition evidence list does not cover every shot boundary"
        )
    if any(
        item.get("production_measurement_evidence") is not True for item in transitions
    ):
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError(
            "one or more accepted transitions lack production measurement evidence"
        )
    return receipt


__all__ = [
    "ProductionGPUQualityRetryError",
    "run_production_continuity_quality_retry_connected_gpu_benchmark",
    "run_production_quality_retry_connected_gpu_benchmark",
]
