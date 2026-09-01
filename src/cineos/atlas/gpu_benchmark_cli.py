"""Production CLI for the connected CINEOS GPU benchmark.

This entrypoint intentionally uses the default CUDA + Diffusers runtime plus the
first-party, hash-bound CINEOS production reference loader. A successful command
therefore means the connected benchmark produced real production GPU execution
evidence without treating approved local identity assets as a test injection.
External pretrained foundation weights remain explicitly identified by the pinned
execution profile.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .foundation_profiles import WAN22_TI2V_5B_PROFILE
from .gpu_connected_benchmark import (
    GPUConnectedBenchmarkReceipt,
    run_connected_gpu_benchmark,
)
from .gpu_persistent_session import PersistentGPUFoundationExecutor
from .native_request import NATIVE_SHOT_SCHEMA, NativeShotRequest
from .production_multi_reference import ProductionReferenceBoardAdapter
from .production_references import ProductionReferenceError, ProductionReferenceLoader


class GPUProductionBenchmarkCLIError(RuntimeError):
    """Raised when production benchmark input or execution is not trustworthy."""


def _request_from_mapping(raw: Mapping[str, Any], *, index: int) -> NativeShotRequest:
    payload = dict(raw)
    supplied_hash = payload.pop("content_hash", "")
    schema = payload.get("schema", NATIVE_SHOT_SCHEMA)
    if schema != NATIVE_SHOT_SCHEMA:
        raise GPUProductionBenchmarkCLIError(
            f"shot {index} uses unsupported native request schema {schema!r}"
        )

    try:
        request = NativeShotRequest(**payload)
    except TypeError as exc:
        raise GPUProductionBenchmarkCLIError(
            f"shot {index} is not a valid native shot request: {exc}"
        ) from exc

    expected_hash = request.refresh_hash()
    if supplied_hash and supplied_hash != expected_hash:
        raise GPUProductionBenchmarkCLIError(
            f"shot {index} content_hash is stale or does not match its payload"
        )
    return request


def load_native_requests(path: str | Path) -> tuple[NativeShotRequest, ...]:
    """Load a 5-10 shot native-request manifest without trusting stored hashes."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GPUProductionBenchmarkCLIError(
            f"cannot read connected-shot request manifest: {source}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise GPUProductionBenchmarkCLIError(
            f"connected-shot request manifest is not valid JSON: {source}"
        ) from exc

    if isinstance(payload, Mapping):
        shots = payload.get("shots")
    else:
        shots = payload
    if not isinstance(shots, Sequence) or isinstance(shots, (str, bytes)):
        raise GPUProductionBenchmarkCLIError(
            "connected-shot request manifest must be a JSON array or contain "
            "a shots array"
        )

    requests: list[NativeShotRequest] = []
    for index, raw in enumerate(shots):
        if not isinstance(raw, Mapping):
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} in request manifest must be a JSON object"
            )
        requests.append(_request_from_mapping(raw, index=index))
    return tuple(requests)


def _production_reference_loader(
    requests: Sequence[NativeShotRequest], reference_manifest: str | Path | None
) -> ProductionReferenceLoader:
    requested_ids = [
        reference_id
        for request in requests
        for reference_id in request.approved_reference_ids
    ]
    if not requested_ids:
        raise GPUProductionBenchmarkCLIError(
            "production connected benchmark requires approved identity references"
        )
    if reference_manifest is None:
        raise GPUProductionBenchmarkCLIError(
            "production connected benchmark requires --reference-manifest so approved "
            "identity assets are hash-bound before GPU execution"
        )
    try:
        loader = ProductionReferenceLoader(reference_manifest)
        loader.validate_reference_ids(requested_ids)
    except ProductionReferenceError as exc:
        raise GPUProductionBenchmarkCLIError(str(exc)) from exc
    return loader


def _production_multi_reference_adapter(
    requests: Sequence[NativeShotRequest],
) -> ProductionReferenceBoardAdapter | None:
    """Enable the audited board only when a shot genuinely declares >1 reference."""

    maximum = max(
        (len(request.approved_reference_ids) for request in requests), default=0
    )
    if maximum <= 1:
        return None
    if maximum > ProductionReferenceBoardAdapter.maximum_references:
        raise GPUProductionBenchmarkCLIError(
            "production connected benchmark supports at most four approved identity "
            "references in one shot with the current audited adapter"
        )
    return ProductionReferenceBoardAdapter()


def run_production_benchmark(
    benchmark_id: str,
    requests: Sequence[NativeShotRequest],
    *,
    output_dir: str | Path,
    reference_manifest: str | Path | None = None,
) -> GPUConnectedBenchmarkReceipt:
    """Run the pinned foundation through one persistent real GPU model session.

    Production connected-shot inference keeps the selected external foundation
    resident across the 5-10 shot sequence. Approved references are resolved only
    through the CINEOS first-party hash-bound manifest loader. Shots with 2-4
    approved identities use the CINEOS-owned deterministic reference-board adapter
    so the foundation's single image slot receives every declared reference rather
    than silently discarding secondary characters.
    """

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    reference_loader = _production_reference_loader(requests, reference_manifest)
    multi_reference_adapter = _production_multi_reference_adapter(requests)
    with PersistentGPUFoundationExecutor(
        WAN22_TI2V_5B_PROFILE,
        output_dir=output_root,
        reference_loader=reference_loader,
        multi_reference_adapter=multi_reference_adapter,
    ) as executor:
        receipt = run_connected_gpu_benchmark(
            benchmark_id,
            requests,
            WAN22_TI2V_5B_PROFILE,
            output_dir=output_root,
            shot_executor=executor,
        )
    if not receipt.production_gpu_evidence:
        raise GPUProductionBenchmarkCLIError(
            "connected benchmark completed without default production CUDA evidence"
        )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real 5-10 shot CINEOS connected GPU benchmark."
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
        help="Stable identifier written into the benchmark evidence manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    requests = load_native_requests(args.requests)
    receipt = run_production_benchmark(
        args.benchmark_id,
        requests,
        output_dir=args.output_dir,
        reference_manifest=args.reference_manifest,
    )
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GPUProductionBenchmarkCLIError",
    "load_native_requests",
    "main",
    "run_production_benchmark",
]
