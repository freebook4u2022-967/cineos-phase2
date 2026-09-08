"""Quality-first production entrypoint for connected GPU film benchmarks.

This module wires CINEOS' production foundation selector into the real self-hosted
CUDA benchmark path. External pretrained video weights remain explicitly identified
by pinned provenance; CINEOS owns the selection, conditioning, continuity, QC and
retry policy rather than the foundation weights themselves.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .artifact_transition_observer import SigLIP2ArtifactTransitionObserver
from .gpu_benchmark_cli import (
    GPUProductionBenchmarkCLIError,
    _production_quality_evaluator,
    _validate_connected_sequence,
    load_native_requests,
)
from .gpu_connected_benchmark import GPUConnectedBenchmarkReceipt
from .gpu_preflight import inspect_cuda_environment
from .gpu_production_quality_retry import (
    ProductionGPUQualityRetryError,
    run_production_continuity_quality_retry_connected_gpu_benchmark as run_production_quality_retry_connected_gpu_benchmark,
)
from .native_request import NativeShotRequest
from .production_foundation_selection import (
    ProductionFoundationSelection,
    ProductionFoundationSelectionError,
    select_strongest_production_foundation,
)
from .transition_quality import ArtifactMeasuredTransitionQualityEvaluator


class _PinnedSigLIP2BoundaryFeatureAdapter:
    """Share the already-loaded pinned shot-QC encoder with seam measurement.

    SigLIP2 remains an external pretrained measurement foundation. This narrow
    adapter deliberately reuses the exact production scorer instance so transition
    QC does not load a second model copy into scarce renderer VRAM.
    """

    def __init__(self, scorer: Any) -> None:
        if getattr(scorer, "semantic_measurement_evidence", False) is not True:
            raise GPUProductionBenchmarkCLIError(
                "production transition QC requires the attested pinned visual scorer"
            )
        if not callable(getattr(scorer, "_pil_frames", None)) or not callable(
            getattr(scorer, "_encode_images", None)
        ):
            raise GPUProductionBenchmarkCLIError(
                "production visual scorer cannot expose boundary feature measurements"
            )
        self.scorer = scorer
        self.semantic_measurement_evidence = True

    def encode_sample_features(self, sample):
        return self.scorer._encode_images(self.scorer._pil_frames(sample))


def _production_transition_evaluator(
    quality_evaluator: Any,
) -> ArtifactMeasuredTransitionQualityEvaluator:
    """Build mandatory artifact-bound seam QC from the same pinned learned scorer."""

    observer = getattr(quality_evaluator, "metric_extractor", None)
    scorer = getattr(observer, "semantic_scorer", None)
    if scorer is None:
        raise GPUProductionBenchmarkCLIError(
            "production quality evaluator is missing its pinned semantic scorer"
        )
    feature_adapter = _PinnedSigLIP2BoundaryFeatureAdapter(scorer)
    transition_observer = SigLIP2ArtifactTransitionObserver(feature_adapter)
    if transition_observer.production_measurement_evidence is not True:
        raise GPUProductionBenchmarkCLIError(
            "production transition observer did not attest measured evidence"
        )
    return ArtifactMeasuredTransitionQualityEvaluator(transition_observer)


def _validate_per_shot_selection_binding(
    receipt: Any,
    selection: ProductionFoundationSelection,
    requests: Sequence[NativeShotRequest],
) -> None:
    """Bind every production shot receipt to the exact selected request and runtime.

    Quality-first production acceptance must not depend on aggregate receipt fields
    alone. Every shot must independently prove the selected profile/origin, exact
    pinned foundation provenance, requested scene/shot identity and request hash,
    plus the CUDA device/dtype chosen from the live quality-first preflight.

    This is intentionally stricter than generic/legacy connected-receipt handling:
    the quality-first production entrypoint only accepts real per-shot evidence and
    fails closed if it is absent, truncated, reordered or replayed.
    """

    shot_receipts = getattr(receipt, "shot_receipts", None)
    if shot_receipts is None:
        raise GPUProductionBenchmarkCLIError(
            "connected benchmark is missing per-shot production evidence"
        )
    if not isinstance(shot_receipts, Sequence) or isinstance(
        shot_receipts, (str, bytes)
    ):
        raise GPUProductionBenchmarkCLIError(
            "connected benchmark shot receipts are malformed"
        )
    if len(shot_receipts) != len(requests):
        raise GPUProductionBenchmarkCLIError(
            "connected benchmark per-shot evidence count does not match requested shots"
        )

    expected_profile = selection.profile
    expected_provenance = expected_profile.provenance
    expected_device = selection.plan.device
    expected_dtype = selection.plan.dtype

    for index, (shot_receipt, request) in enumerate(zip(shot_receipts, requests)):
        if getattr(shot_receipt, "profile_id", None) != expected_profile.profile_id:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} profile does not match quality-first selection"
            )
        if getattr(shot_receipt, "origin", None) != expected_profile.origin:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} origin does not match quality-first selection"
            )

        result = getattr(shot_receipt, "result", None)
        if result is None or getattr(result, "foundation", None) != expected_provenance:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} foundation provenance does not match quality-first selection"
            )
        if getattr(result, "scene_id", None) != request.scene_id:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} scene identity does not match requested shot"
            )
        if getattr(result, "shot_id", None) != request.shot_id:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} shot identity does not match requested shot"
            )
        if getattr(result, "request_hash", None) != request.content_hash:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} request hash does not match requested shot"
            )

        execution_plan = getattr(shot_receipt, "execution_plan", None)
        if execution_plan is None:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} is missing GPU execution-plan evidence"
            )
        if getattr(execution_plan, "device", None) != expected_device:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} CUDA device does not match quality-first selection"
            )
        if getattr(execution_plan, "dtype", None) != expected_dtype:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} dtype does not match quality-first selection"
            )

        runtime = getattr(shot_receipt, "runtime_provenance", None)
        if not isinstance(runtime, dict):
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} is missing runtime provenance"
            )
        if runtime.get("cuda_device") != expected_device:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} runtime CUDA device does not match quality-first selection"
            )
        if runtime.get("dtype") != expected_dtype:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} runtime dtype does not match quality-first selection"
            )


def run_quality_first_production_benchmark(
    benchmark_id: str,
    requests: Sequence[NativeShotRequest],
    *,
    output_dir: str | Path,
    reference_manifest: str | Path | None = None,
    continuity_identity_refresh: bool = False,
    devices=None,
) -> GPUConnectedBenchmarkReceipt:
    """Run the strongest approved pinned foundation that the live runner can fit.

    ``devices`` is injectable only to make selection regression-testable without CUDA;
    production callers omit it so the live CUDA environment is inspected immediately
    before model execution.
    """

    if not isinstance(continuity_identity_refresh, bool):
        raise TypeError("continuity_identity_refresh must be a bool")
    _validate_connected_sequence(requests)

    if devices is None:
        devices = inspect_cuda_environment()
    try:
        selection = select_strongest_production_foundation(tuple(devices), requests)
    except ProductionFoundationSelectionError as exc:
        raise GPUProductionBenchmarkCLIError(str(exc)) from exc

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    quality_evaluator = _production_quality_evaluator(requests, reference_manifest)
    transition_evaluator = _production_transition_evaluator(quality_evaluator)

    selection_manifest = output_root / "foundation-selection.json"
    selection_manifest.write_text(
        json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        receipt = run_production_quality_retry_connected_gpu_benchmark(
            benchmark_id,
            requests,
            selection.profile,
            output_dir=output_root,
            quality_evaluator=quality_evaluator,
            transition_evaluator=transition_evaluator,
            reference_manifest=reference_manifest,
            continuity_identity_refresh=continuity_identity_refresh,
        )
    except ProductionGPUQualityRetryError as exc:
        raise GPUProductionBenchmarkCLIError(str(exc)) from exc

    if receipt.profile_id != selection.profile.profile_id:
        raise GPUProductionBenchmarkCLIError(
            "connected benchmark receipt profile does not match quality-first selection: "
            f"selected={selection.profile.profile_id!r} receipt={receipt.profile_id!r}"
        )
    if receipt.origin != selection.profile.origin:
        raise GPUProductionBenchmarkCLIError(
            "connected benchmark receipt origin does not match quality-first selection: "
            f"selected={selection.profile.origin!r} receipt={receipt.origin!r}"
        )
    _validate_per_shot_selection_binding(receipt, selection, requests)
    if not receipt.production_gpu_evidence:
        raise GPUProductionBenchmarkCLIError(
            "connected benchmark completed without default production CUDA evidence"
        )
    if not receipt.production_quality_evidence:
        raise GPUProductionBenchmarkCLIError(
            "connected benchmark completed without artifact-bound production QC evidence"
        )
    if receipt.evidence_tier != "production-gpu-quality-gated":
        raise GPUProductionBenchmarkCLIError(
            "connected benchmark did not reach production-gpu-quality-gated evidence tier"
        )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the strongest safely executable pinned video foundation on a real "
            "5-10 shot CINEOS connected GPU benchmark with mandatory learned QC."
        )
    )
    parser.add_argument("--requests", required=True, help="Native shot JSON manifest")
    parser.add_argument(
        "--reference-manifest",
        required=True,
        help="Hash-pinned approved reference JSON manifest",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Benchmark artifact directory"
    )
    parser.add_argument(
        "--benchmark-id",
        default="cineos-connected-production",
        help="Stable identifier written into benchmark evidence",
    )
    parser.add_argument(
        "--continuity-identity-refresh",
        action="store_true",
        help=(
            "Run the experimental predecessor-frame + fresh-reference continuity "
            "conditioning strategy."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    requests = load_native_requests(args.requests)
    receipt = run_quality_first_production_benchmark(
        args.benchmark_id,
        requests,
        output_dir=args.output_dir,
        reference_manifest=args.reference_manifest,
        continuity_identity_refresh=args.continuity_identity_refresh,
    )
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_quality_first_production_benchmark"]
