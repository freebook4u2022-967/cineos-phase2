"""Fail-fast validation for real connected-production GPU inputs.

This module intentionally runs before heavyweight foundation/QC model acquisition on
self-hosted GPU workers. It reuses the same connected-shot and production-reference
contracts as the execution path so malformed continuity graphs or stale approved
identity assets cannot consume scarce model-download/load time.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .gpu_benchmark_cli import (
    GPUProductionBenchmarkCLIError,
    _production_multi_reference_adapter,
    _production_reference_loader,
    load_native_requests,
)
from .gpu_connected_benchmark import GPUConnectedBenchmarkError, _validate_requests
from .native_request import NativeShotRequest
from .production_references import ProductionReferenceError


class ProductionInputPreflightError(RuntimeError):
    """Raised when connected-production inputs cannot produce trustworthy evidence."""


def preflight_production_inputs(
    requests: Sequence[NativeShotRequest],
    reference_manifest: str | Path,
) -> dict[str, object]:
    """Validate the complete connected request/reference boundary without model loads."""

    request_sequence = tuple(requests)
    try:
        _validate_requests(request_sequence)
        _production_multi_reference_adapter(request_sequence)
        loader = _production_reference_loader(request_sequence, reference_manifest)

        requested_reference_ids = tuple(
            dict.fromkeys(
                reference_id
                for request in request_sequence
                for reference_id in request.approved_reference_ids
            )
        )
        # validate_reference_ids already verifies presence + SHA-256. Decode every
        # distinct asset too, so an approved-but-corrupt image fails before model IO.
        for reference_id in requested_reference_ids:
            loader(reference_id)
    except (
        GPUConnectedBenchmarkError,
        GPUProductionBenchmarkCLIError,
        ProductionReferenceError,
    ) as exc:
        raise ProductionInputPreflightError(str(exc)) from exc

    return {
        "schema": "cineos-production-input-preflight/0.1",
        "shot_count": len(request_sequence),
        "reference_count": len(requested_reference_ids),
        "reference_manifest_sha256": loader.manifest_sha256,
        "validated": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate connected CINEOS production requests and hash-pinned identity "
            "assets without loading video or QC model weights."
        )
    )
    parser.add_argument("--requests", required=True)
    parser.add_argument("--reference-manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        requests = load_native_requests(args.requests)
        result = preflight_production_inputs(requests, args.reference_manifest)
    except (GPUProductionBenchmarkCLIError, ProductionInputPreflightError) as exc:
        raise SystemExit(f"production input preflight failed: {exc}") from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ProductionInputPreflightError",
    "main",
    "preflight_production_inputs",
]
