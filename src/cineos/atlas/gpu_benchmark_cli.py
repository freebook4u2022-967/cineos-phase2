"""Production CLI for the connected CINEOS GPU benchmark.

This entrypoint intentionally uses the default CUDA + Diffusers runtime plus the
first-party, hash-bound CINEOS production reference loader. Production execution is
also required to pass the artifact-bound learned visual-QC and reject/rerender gate;
a render is not accepted merely because CUDA inference completed. External pretrained
foundation and QC weights remain explicitly identified by their pinned provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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


REQUIRED_COMPETITIVE_CHALLENGES = frozenset(
    {
        "identity_consistency",
        "multi_character_interaction",
        "hands_anatomy",
        "walking_running",
        "dialogue_lip_sync",
        "object_interaction",
        "fast_camera_movement",
        "lighting_changes",
        "physics",
    }
)
COMPETITIVE_CHALLENGE_METADATA_KEY = "competitive_challenges"


def _validate_connected_shot_count(requests: Sequence[NativeShotRequest]) -> None:
    """Fail closed unless the production benchmark contains exactly 5-10 shots."""

    shot_count = len(requests)
    if not 5 <= shot_count <= 10:
        raise GPUProductionBenchmarkCLIError(
            "production connected benchmark requires 5-10 shots; "
            f"received {shot_count}"
        )


def _expected_request_hash(request: NativeShotRequest) -> str:
    """Compute the canonical native-request hash without mutating caller state."""

    payload = json.dumps(
        request.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_request_hashes(requests: Sequence[NativeShotRequest]) -> None:
    """Require every direct production request to be hash-bound to its live payload."""

    for index, request in enumerate(requests):
        supplied = request.content_hash
        if not isinstance(supplied, str) or len(supplied) != 64:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} requires a canonical 64-character content_hash before "
                "production execution"
            )
        expected = _expected_request_hash(request)
        if supplied != expected:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} content_hash is stale or does not match its live payload"
            )


def _continuity_predecessor(request: NativeShotRequest, *, index: int) -> str | None:
    """Return the declared predecessor while preserving the legacy field alias."""

    continuity = request.continuity
    current = continuity.get("previous_shot_id")
    legacy = continuity.get("previous_shot")
    if current not in (None, "") and legacy not in (None, "") and current != legacy:
        raise GPUProductionBenchmarkCLIError(
            f"shot {index} has conflicting previous_shot_id/previous_shot continuity"
        )
    predecessor = current if current not in (None, "") else legacy
    if predecessor is None or predecessor == "":
        return None
    if not isinstance(predecessor, str):
        raise GPUProductionBenchmarkCLIError(
            f"shot {index} continuity predecessor must be a string or null"
        )
    return predecessor


def _challenge_tags(request: NativeShotRequest, *, index: int) -> frozenset[str]:
    raw = request.metadata.get(COMPETITIVE_CHALLENGE_METADATA_KEY)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise GPUProductionBenchmarkCLIError(
            f"shot {index} must declare a non-empty "
            f"{COMPETITIVE_CHALLENGE_METADATA_KEY!r} sequence"
        )

    tags: set[str] = set()
    for challenge in raw:
        if not isinstance(challenge, str) or not challenge.strip():
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} contains an invalid competitive challenge tag"
            )
        normalized = challenge.strip()
        if normalized not in REQUIRED_COMPETITIVE_CHALLENGES:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} declares unknown competitive challenge {normalized!r}"
            )
        tags.add(normalized)
    return frozenset(tags)


def _validate_challenge_structure(
    requests: Sequence[NativeShotRequest], challenge_tags: Sequence[frozenset[str]]
) -> None:
    """Bind objectively checkable challenge claims to request structure.

    These checks do not assert that a rendered artifact solves a challenge. They only
    prevent impossible or vacuous benchmark declarations from reaching expensive GPU
    execution. Learned/artifact QC remains responsible for judging visual success.
    """

    reference_occurrences = Counter(
        reference_id
        for request in requests
        for reference_id in set(request.approved_reference_ids)
    )

    for index, (request, tags) in enumerate(zip(requests, challenge_tags, strict=True)):
        if "multi_character_interaction" in tags:
            if len(request.characters) < 2 or len(set(request.approved_reference_ids)) < 2:
                raise GPUProductionBenchmarkCLIError(
                    f"shot {index} declares multi_character_interaction but does not "
                    "contain at least two characters with two distinct approved "
                    "identity references"
                )

        if "object_interaction" in tags and not request.props:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} declares object_interaction but contains no prop "
                "conditioning"
            )

        if "dialogue_lip_sync" in tags:
            dialogue_timing = request.performance.get("dialogue_timing")
            if not isinstance(dialogue_timing, Sequence) or isinstance(
                dialogue_timing, (str, bytes)
            ) or not dialogue_timing:
                raise GPUProductionBenchmarkCLIError(
                    f"shot {index} declares dialogue_lip_sync but contains no "
                    "dialogue_timing performance evidence"
                )

        if "identity_consistency" in tags:
            persistent_ids = {
                reference_id
                for reference_id in request.approved_reference_ids
                if reference_occurrences[reference_id] >= 2
            }
            if not persistent_ids:
                raise GPUProductionBenchmarkCLIError(
                    f"shot {index} declares identity_consistency but none of its "
                    "approved identity references persists into another connected shot"
                )


def _validate_competitive_challenge_coverage(
    requests: Sequence[NativeShotRequest],
) -> None:
    """Require explicit coverage of every mandatory difficult production stressor.

    The tags are CINEOS benchmark declarations, not claims that the renderer solved a
    challenge. Actual success remains determined by artifact-bound QC and benchmark
    metrics after real inference. Requiring declarations here prevents an easy 5-10
    shot sequence from being presented as evidence for the competitive release gate.
    """

    per_shot_tags = tuple(
        _challenge_tags(request, index=index)
        for index, request in enumerate(requests)
    )
    covered = set().union(*per_shot_tags)
    missing = sorted(REQUIRED_COMPETITIVE_CHALLENGES - covered)
    if missing:
        raise GPUProductionBenchmarkCLIError(
            "production connected benchmark does not cover all mandatory competitive "
            f"challenges; missing: {', '.join(missing)}"
        )
    _validate_challenge_structure(requests, per_shot_tags)


def _validate_connected_sequence(requests: Sequence[NativeShotRequest]) -> None:
    """Require an ordered, hash-bound predecessor chain before GPU/model loading."""

    _validate_connected_shot_count(requests)
    _validate_request_hashes(requests)
    shot_ids = [request.shot_id for request in requests]
    if any(not isinstance(shot_id, str) or not shot_id.strip() for shot_id in shot_ids):
        raise GPUProductionBenchmarkCLIError(
            "production connected benchmark requires non-empty string shot_id values"
        )
    if len(set(shot_ids)) != len(shot_ids):
        raise GPUProductionBenchmarkCLIError(
            "production connected benchmark requires unique shot_id values"
        )

    first_predecessor = _continuity_predecessor(requests[0], index=0)
    if first_predecessor is not None:
        raise GPUProductionBenchmarkCLIError(
            "first shot in production connected benchmark must not declare a predecessor"
        )

    for index, request in enumerate(requests[1:], start=1):
        expected = shot_ids[index - 1]
        predecessor = _continuity_predecessor(request, index=index)
        if predecessor != expected:
            raise GPUProductionBenchmarkCLIError(
                "production connected benchmark continuity is not a contiguous ordered "
                f"chain: shot {index} ({request.shot_id!r}) must reference {expected!r} "
                f"as its predecessor, received {predecessor!r}"
            )

    _validate_competitive_challenge_coverage(requests)


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
    """Load a 5-10 shot connected native-request manifest without trusting hashes."""

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
    loaded = tuple(requests)
    _validate_connected_sequence(loaded)
    return loaded


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
    _validate_connected_sequence(requests)
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
            "connected benchmark completed without artifact-bound production QC "
            "evidence"
        )
    if receipt.evidence_tier != "production-gpu-quality-gated":
        raise GPUProductionBenchmarkCLIError(
            "connected benchmark did not reach production-gpu-quality-gated "
            "evidence tier"
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
    "COMPETITIVE_CHALLENGE_METADATA_KEY",
    "GPUProductionBenchmarkCLIError",
    "REQUIRED_COMPETITIVE_CHALLENGES",
    "load_native_requests",
    "main",
    "run_production_benchmark",
]
