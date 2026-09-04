"""Production CLI for the connected CINEOS GPU benchmark.

This entrypoint intentionally uses the default CUDA + Diffusers runtime plus the
first-party, hash-bound CINEOS production reference loader. Production execution is
also required to pass the artifact-bound learned visual-QC and reject/rerender gate;
a render is not accepted merely because CUDA inference completed. External pretrained
foundation and QC weights remain explicitly identified by their pinned provenance.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .artifact_video_observer import ArtifactVideoMetricObserver
from .foundation_profiles import WAN22_TI2V_5B_PROFILE
from .gpu_connected_benchmark import GPUConnectedBenchmarkReceipt
from .gpu_production_quality_retry import (
    ProductionGPUQualityRetryError,
    run_production_quality_retry_connected_gpu_benchmark,
)
from .native_request import NATIVE_SHOT_SCHEMA, NativeShotRequest
from .production_multi_reference import ProductionReferenceBoardAdapter
from .production_references import ProductionReferenceError, ProductionReferenceLoader
from .sequence_quality import ArtifactMeasuredSequenceQualityEvaluator
from .siglip2_video_scorer import SigLIP2FeatureVideoScorer, SigLIP2VideoScorerError


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
        unique_requested_ids = tuple(dict.fromkeys(requested_ids))
        ids_by_hash: dict[str, list[str]] = {}
        for reference_id in unique_requested_ids:
            digest = loader.reference_sha256(reference_id)
            ids_by_hash.setdefault(digest, []).append(reference_id)
        duplicate_content_groups = [
            reference_ids
            for reference_ids in ids_by_hash.values()
            if len(reference_ids) > 1
        ]
        if duplicate_content_groups:
            aliases = "; ".join(
                ", ".join(reference_ids) for reference_ids in duplicate_content_groups
            )
            raise ProductionReferenceError(
                "production reference ids must resolve to distinct approved content; "
                f"duplicate SHA-256 payloads: {aliases}"
            )
    except ProductionReferenceError as exc:
        raise GPUProductionBenchmarkCLIError(str(exc)) from exc
    return loader


def _production_multi_reference_adapter(
    requests: Sequence[NativeShotRequest],
) -> ProductionReferenceBoardAdapter | None:
    """Validate current audited multi-reference capacity before model loading."""

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


def _production_quality_evaluator(
    requests: Sequence[NativeShotRequest],
    reference_manifest: str | Path | None,
) -> ArtifactMeasuredSequenceQualityEvaluator:
    """Build the pinned learned observer used by the production reject/rerender gate.

    SigLIP2 is an external Apache-2.0 pretrained QC foundation. CINEOS owns the
    artifact binding, policy and retry decision, not the SigLIP2 model weights.
    ``local_files_only`` is enforced inside the scorer so the workflow must prefetch
    the exact pinned revision before production inference begins.
    """

    loader = _production_reference_loader(requests, reference_manifest)
    _production_multi_reference_adapter(requests)
    try:
        scorer = SigLIP2FeatureVideoScorer(loader, device="cuda")
    except SigLIP2VideoScorerError as exc:
        raise GPUProductionBenchmarkCLIError(
            f"cannot initialize pinned production visual QC: {exc}"
        ) from exc
    observer = ArtifactVideoMetricObserver(scorer)
    if observer.production_measurement_evidence is not True:
        raise GPUProductionBenchmarkCLIError(
            "production visual QC scorer did not attest measured semantic evidence"
        )
    return ArtifactMeasuredSequenceQualityEvaluator(observer)


def run_production_benchmark(
    benchmark_id: str,
    requests: Sequence[NativeShotRequest],
    *,
    output_dir: str | Path,
    reference_manifest: str | Path | None = None,
    continuity_identity_refresh: bool = False,
) -> GPUConnectedBenchmarkReceipt:
    """Run pinned foundation inference behind mandatory learned visual QC + retry.

    The external video foundation remains explicitly identified by the immutable
    execution profile. Approved references are hash-bound through CINEOS production
    loaders. Each rendered shot must then pass artifact-bound measured QC; rejected
    attempts receive auditable CINEOS retry directives and deterministic seed changes
    before another real GPU render is attempted. This path deliberately does not
    treat successful inference alone as production-quality evidence.
    """

    if not isinstance(continuity_identity_refresh, bool):
        raise TypeError("continuity_identity_refresh must be a bool")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    quality_evaluator = _production_quality_evaluator(requests, reference_manifest)
    try:
        receipt = run_production_quality_retry_connected_gpu_benchmark(
            benchmark_id,
            requests,
            WAN22_TI2V_5B_PROFILE,
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
            "Run a real 5-10 shot CINEOS connected GPU benchmark with mandatory "
            "artifact-bound learned QC and reject/rerender."
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
        help="Stable identifier written into the benchmark evidence manifest",
    )
    parser.add_argument(
        "--continuity-identity-refresh",
        action="store_true",
        help=(
            "Run the experimental CINEOS predecessor-frame + fresh-reference "
            "conditioning strategy for a measured GPU A/B candidate."
        ),
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
        continuity_identity_refresh=args.continuity_identity_refresh,
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
