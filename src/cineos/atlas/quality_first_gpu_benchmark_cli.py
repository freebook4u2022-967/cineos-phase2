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
    run_production_quality_retry_connected_gpu_benchmark,
)
from .native_request import NativeShotRequest
from .production_foundation_selection import (
    ProductionFoundationSelectionError,
    select_strongest_production_foundation,
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
            reference_manifest=reference_manifest,
            continuity_identity_refresh=continuity_identity_refresh,
        )
    except ProductionGPUQualityRetryError as exc:
        raise GPUProductionBenchmarkCLIError(str(exc)) from exc

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
