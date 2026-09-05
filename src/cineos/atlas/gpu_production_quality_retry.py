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

from .connected_continuity_evidence import (
    ConnectedContinuityEvidenceError,
    validate_connected_visual_continuity,
)
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
from .production_continuity_identity import compose_continuity_identity_board
from .production_multi_reference import ProductionReferenceBoardAdapter
from .production_references import ProductionReferenceError, ProductionReferenceLoader
from .quality_retry import QualityRetryPolicy
from .sequence_quality import ArtifactMeasuredSequenceQualityEvaluator
from .transition_quality import ArtifactMeasuredTransitionQualityEvaluator


class ProductionGPUQualityRetryError(GPUQualityRetryBenchmarkError):
    """Raised when a quality-retry run lacks production GPU or measurement evidence."""


_INJECTED_RUNTIME_KWARGS = frozenset(
    {
        "torch_module",
        "reference_loader",
        "multi_reference_adapter",
        "continuity_identity_adapter",
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


def _production_identity_runtime(
    requests: Sequence[NativeShotRequest],
    *,
    reference_manifest: str | Path | None,
    continuity_identity_refresh: bool,
) -> dict[str, Any]:
    """Build only audited first-party identity-conditioning boundaries.

    The public production API accepts paths and a boolean strategy selector rather
    than arbitrary loader/adapter callables. This keeps the runtime reproducible and
    prevents test or borrowed conditioning code from being promoted to production
    evidence while allowing both baseline and experimental A/B runs to use the exact
    same approved assets.
    """

    if not isinstance(continuity_identity_refresh, bool):
        raise TypeError("continuity_identity_refresh must be a bool")

    requested_ids = [
        reference_id
        for request in requests
        for reference_id in request.approved_reference_ids
    ]
    if not requested_ids:
        raise ProductionGPUQualityRetryError(
            "production quality benchmark requires approved identity references"
        )
    if reference_manifest is None:
        raise ProductionGPUQualityRetryError(
            "production quality benchmark requires a hash-pinned reference manifest"
        )

    try:
        reference_loader = ProductionReferenceLoader(reference_manifest)
        reference_loader.validate_reference_ids(requested_ids)
    except ProductionReferenceError as exc:
        raise ProductionGPUQualityRetryError(str(exc)) from exc

    maximum_references = max(
        (len(request.approved_reference_ids) for request in requests),
        default=0,
    )
    multi_reference_adapter: ProductionReferenceBoardAdapter | None = None
    if maximum_references > ProductionReferenceBoardAdapter.maximum_references:
        raise ProductionGPUQualityRetryError(
            "production quality benchmark supports at most four approved identity "
            "references per shot with the current audited adapter"
        )
    if maximum_references > 1:
        multi_reference_adapter = ProductionReferenceBoardAdapter()

    runtime: dict[str, Any] = {"reference_loader": reference_loader}
    if multi_reference_adapter is not None:
        runtime["multi_reference_adapter"] = multi_reference_adapter
    if continuity_identity_refresh:
        runtime["continuity_identity_adapter"] = compose_continuity_identity_board
    return runtime


def run_production_quality_retry_connected_gpu_benchmark(
    benchmark_id: str,
    requests: Sequence[NativeShotRequest],
    profile: FoundationExecutionProfile,
    *,
    output_dir: str | Path,
    quality_evaluator: QualityEvaluator,
    transition_evaluator: TransitionEvaluator | None = None,
    retry_policy: QualityRetryPolicy | None = None,
    reference_manifest: str | Path | None = None,
    continuity_identity_refresh: bool = False,
    shot_executor: ShotExecutor = execute_foundation_gpu_shot,
    shot_executor_kwargs: dict[str, Any] | None = None,
) -> GPUConnectedBenchmarkReceipt:
    """Run 5-10 shots requiring real GPU and artifact-bound shot QC evidence.

    Approved identity assets are resolved internally through the hash-pinned CINEOS
    production manifest loader. Multi-reference composition and the optional
    experimental continuity + fresh-reference strategy are also selected internally;
    callers cannot inject substitute conditioning code into production evidence.

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
    identity_runtime = _production_identity_runtime(
        requests,
        reference_manifest=reference_manifest,
        continuity_identity_refresh=continuity_identity_refresh,
    )
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    persistent_kwargs: dict[str, Any] = dict(identity_runtime)
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
            "production quality evidence required, but one or more accepted shots lack "
            "artifact-bound measured QC evidence"
        )
    if receipt.evidence_tier != "production-gpu-quality-gated":
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError(
            "production benchmark did not reach the production-gpu-quality-gated "
            "evidence tier"
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
    reference_manifest: str | Path | None = None,
    continuity_identity_refresh: bool = False,
    shot_executor: ShotExecutor = execute_foundation_gpu_shot,
    shot_executor_kwargs: dict[str, Any] | None = None,
) -> GPUConnectedBenchmarkReceipt:
    """Run the strict production sequence gate with visual lineage and seam QC."""

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
        reference_manifest=reference_manifest,
        continuity_identity_refresh=continuity_identity_refresh,
        shot_executor=shot_executor,
        shot_executor_kwargs=shot_executor_kwargs,
    )
    manifest = Path(receipt.manifest_path)
    try:
        validate_connected_visual_continuity(receipt.shot_receipts)
    except ConnectedContinuityEvidenceError as exc:
        _remove_stale_manifest(manifest)
        raise ProductionGPUQualityRetryError(
            "production continuity lacks artifact-bound terminal-frame lineage"
        ) from exc
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
