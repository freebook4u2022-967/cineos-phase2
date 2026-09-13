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
from .artifact_video_observer import ArtifactVideoMetricObserver
from .composite_semantic_scorer import CompositeSemanticVideoScorer
from .gpu_benchmark_cli import (
    GPUProductionBenchmarkCLIError,
    _production_quality_evaluator,
    _validate_connected_sequence,
    load_native_requests,
)
from .gpu_connected_benchmark import GPUConnectedBenchmarkReceipt
from .gpu_preflight import inspect_cuda_environment
from .gpu_production_quality_retry import ProductionGPUQualityRetryError
from .gpu_production_quality_retry import (
    run_production_continuity_quality_retry_connected_gpu_benchmark as run_production_quality_retry_connected_gpu_benchmark,
)
from .native_request import NativeShotRequest
from .production_benchmark_attestation import (
    ProductionBenchmarkAttestationError,
    remove_stale_quality_first_attestation,
    write_quality_first_production_attestation,
)
from .production_foundation_selection import (
    ProductionFoundationSelection,
    ProductionFoundationSelectionError,
    select_strongest_production_foundation,
)
from .production_references import ProductionReferenceError, ProductionReferenceLoader
from .qwen25vl_semantic_judge import Qwen25VLSemanticJudge
from .sequence_quality import ArtifactMeasuredSequenceQualityEvaluator
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


def _production_semantic_quality_evaluator(
    requests: Sequence[NativeShotRequest],
    reference_manifest: str | Path | None,
) -> Any:
    """Compose core identity/motion QC with the pinned difficult-case visual judge.

    The generic production CLI intentionally keeps its historical SigLIP2-only
    behavior. The quality-first release path adds Qwen2.5-VL as an external,
    provenance-preserving specialist so anatomy, interaction, locomotion, camera,
    lighting and physics measurements participate in the same reject/rerender gate.
    Dialogue lip-sync remains owned by the independent audio/visual specialist path.

    A nonstandard evaluator returned by an injected private factory is passed through
    unchanged. This preserves the historical test/integration injection seam; the
    real production factory returns ArtifactMeasuredSequenceQualityEvaluator and is
    therefore always upgraded to the specialist-composed observer.
    """

    base = _production_quality_evaluator(requests, reference_manifest)
    if not isinstance(base, ArtifactMeasuredSequenceQualityEvaluator):
        return base
    observer = getattr(base, "metric_extractor", None)
    primary = getattr(observer, "semantic_scorer", None)
    if primary is None:
        raise GPUProductionBenchmarkCLIError(
            "production quality evaluator is missing its pinned primary semantic scorer"
        )
    try:
        composite = CompositeSemanticVideoScorer(
            primary,
            specialists=(Qwen25VLSemanticJudge(),),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise GPUProductionBenchmarkCLIError(
            f"cannot initialize quality-first specialist semantic QC: {exc}"
        ) from exc
    sampler = getattr(observer, "sampler", None)
    observer_id = getattr(observer, "observer_id", "cineos-artifact-video-observer/0.1")
    composed_observer = ArtifactVideoMetricObserver(
        composite,
        sampler=sampler,
        observer_id=observer_id,
    )
    if composed_observer.production_measurement_evidence is not True:
        raise GPUProductionBenchmarkCLIError(
            "quality-first specialist semantic QC did not attest measured evidence"
        )
    return ArtifactMeasuredSequenceQualityEvaluator(composed_observer)


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
    # Quality-first shot QC composes specialist scorers around the SigLIP2 primary.
    # Seam QC still needs the primary feature encoder directly; unwrapping it here
    # avoids loading a second SigLIP2 copy and does not attribute Qwen metrics to the
    # transition observer.
    scorer = getattr(scorer, "primary", scorer)
    feature_adapter = _PinnedSigLIP2BoundaryFeatureAdapter(scorer)
    transition_observer = SigLIP2ArtifactTransitionObserver(feature_adapter)
    if transition_observer.production_measurement_evidence is not True:
        raise GPUProductionBenchmarkCLIError(
            "production transition observer did not attest measured evidence"
        )
    return ArtifactMeasuredTransitionQualityEvaluator(transition_observer)


def _is_sha256_hexdigest(value: Any) -> bool:
    """Return whether ``value`` is a canonical SHA-256 hexadecimal digest."""

    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _expected_character_reference_bindings(
    request: NativeShotRequest,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the renderer-canonical character-to-reference ownership contract."""

    bindings: list[tuple[str, tuple[str, ...]]] = []
    for index, character in enumerate(getattr(request, "characters", ())):
        if not isinstance(character, dict):
            continue
        character_id = character.get("character_uuid", f"index:{index}")
        if not isinstance(character_id, str) or not character_id.strip():
            character_id = f"index:{index}"
        else:
            character_id = character_id.strip()
        raw_ids = character.get("approved_reference_ids", [])
        if isinstance(raw_ids, (list, tuple)) and raw_ids:
            bindings.append((character_id, tuple(raw_ids)))
    return tuple(bindings)


def _validate_character_reference_binding(
    conditioning: dict[str, Any],
    request: NativeShotRequest,
    *,
    shot_index: int,
) -> None:
    """Bind receipt identity ownership to the exact CINEOS character contract."""

    expected_bindings = _expected_character_reference_bindings(request)
    reported = conditioning.get("consumed_character_reference_ids")
    if not expected_bindings:
        if reported not in (None, []):
            raise GPUProductionBenchmarkCLIError(
                f"shot {shot_index} reports character reference ownership for an unmapped request"
            )
        return
    if not isinstance(reported, (list, tuple)):
        raise GPUProductionBenchmarkCLIError(
            f"shot {shot_index} is missing character-to-reference conditioning provenance"
        )

    normalized: list[tuple[str, tuple[str, ...]]] = []
    for binding_index, binding in enumerate(reported):
        if not isinstance(binding, dict):
            raise GPUProductionBenchmarkCLIError(
                f"shot {shot_index} character reference binding {binding_index} is malformed"
            )
        character_id = binding.get("character_uuid")
        reference_ids = binding.get("reference_ids")
        if (
            not isinstance(character_id, str)
            or not character_id.strip()
            or not isinstance(reference_ids, (list, tuple))
            or any(
                not isinstance(reference_id, str) or not reference_id.strip()
                for reference_id in reference_ids
            )
        ):
            raise GPUProductionBenchmarkCLIError(
                f"shot {shot_index} character reference binding {binding_index} is malformed"
            )
        normalized.append((character_id.strip(), tuple(reference_ids)))

    if tuple(normalized) != expected_bindings:
        raise GPUProductionBenchmarkCLIError(
            f"shot {shot_index} character-to-reference conditioning does not match the approved ownership contract"
        )


def _validate_conditioning_binding(
    result: Any,
    request: NativeShotRequest,
    *,
    shot_index: int,
    expected_reference_hashes: Sequence[str] | None = None,
) -> None:
    """Bind production conditioning evidence to the exact approved reference board.

    ProductionDiffusersVideoResult exposes ``conditioning_provenance``. When that
    production field is present, quality-first acceptance fails closed unless it
    proves that every approved reference was actually consumed in request order and
    carries renderer-computed content fingerprints for both the consumed references
    and the exact image supplied to the external foundation. When immutable manifest
    digests are supplied by the production entrypoint, the renderer-computed hashes
    must also match those approved bytes exactly; a correct reference ID alone is not
    sufficient evidence of identity conditioning. Character-local identity ownership
    is independently rebound here so a receipt cannot retain the global reference set
    while swapping which character consumed which approved identity source.

    Generic/legacy result objects that predate this production evidence field retain
    their historical compatibility outside the real production renderer boundary.
    """

    if not hasattr(result, "conditioning_provenance"):
        return

    conditioning = getattr(result, "conditioning_provenance", None)
    expected_references = tuple(request.approved_reference_ids)
    if not expected_references:
        if conditioning not in (None, {}):
            raise GPUProductionBenchmarkCLIError(
                f"shot {shot_index} reports conditioning for a request with no approved references"
            )
        return

    if not isinstance(conditioning, dict):
        raise GPUProductionBenchmarkCLIError(
            f"shot {shot_index} is missing production conditioning provenance"
        )

    consumed = conditioning.get("consumed_reference_ids")
    if (
        not isinstance(consumed, (list, tuple))
        or tuple(consumed) != expected_references
    ):
        raise GPUProductionBenchmarkCLIError(
            f"shot {shot_index} conditioning references do not match the approved reference board"
        )

    _validate_character_reference_binding(conditioning, request, shot_index=shot_index)

    consumed_hashes = conditioning.get("consumed_reference_sha256")
    if (
        not isinstance(consumed_hashes, (list, tuple))
        or len(consumed_hashes) != len(expected_references)
        or any(not _is_sha256_hexdigest(digest) for digest in consumed_hashes)
    ):
        raise GPUProductionBenchmarkCLIError(
            f"shot {shot_index} conditioning reference fingerprints are missing or invalid"
        )
    if len(set(consumed_hashes)) != len(consumed_hashes):
        raise GPUProductionBenchmarkCLIError(
            f"shot {shot_index} conditioning references resolve to duplicate consumed content"
        )

    if expected_reference_hashes is not None:
        approved_hashes = tuple(expected_reference_hashes)
        if len(approved_hashes) != len(expected_references) or any(
            not _is_sha256_hexdigest(digest) for digest in approved_hashes
        ):
            raise GPUProductionBenchmarkCLIError(
                f"shot {shot_index} approved reference fingerprints are missing or invalid"
            )
        if tuple(consumed_hashes) != approved_hashes:
            raise GPUProductionBenchmarkCLIError(
                f"shot {shot_index} consumed reference content does not match approved manifest bytes"
            )

    conditioning_image_hash = conditioning.get("conditioning_image_sha256")
    if not _is_sha256_hexdigest(conditioning_image_hash):
        raise GPUProductionBenchmarkCLIError(
            f"shot {shot_index} conditioning image fingerprint is missing or invalid"
        )

    mode = conditioning.get("mode")
    if len(expected_references) == 1:
        if mode != "single_reference":
            raise GPUProductionBenchmarkCLIError(
                f"shot {shot_index} single-reference conditioning mode is invalid"
            )
        if conditioning_image_hash != consumed_hashes[0]:
            raise GPUProductionBenchmarkCLIError(
                f"shot {shot_index} single-reference conditioning image does not match consumed content"
            )
        return

    if mode != "multi_reference_adapter":
        raise GPUProductionBenchmarkCLIError(
            f"shot {shot_index} multi-reference conditioning mode is invalid"
        )
    if conditioning_image_hash in consumed_hashes:
        raise GPUProductionBenchmarkCLIError(
            f"shot {shot_index} multi-reference adapter returned an unchanged source reference"
        )
    adapter_id = conditioning.get("adapter_id")
    adapter_version = conditioning.get("adapter_version")
    if not isinstance(adapter_id, str) or not adapter_id.strip():
        raise GPUProductionBenchmarkCLIError(
            f"shot {shot_index} multi-reference conditioning is missing adapter provenance"
        )
    if not isinstance(adapter_version, str) or not adapter_version.strip():
        raise GPUProductionBenchmarkCLIError(
            f"shot {shot_index} multi-reference conditioning is missing adapter provenance"
        )


def _validate_per_shot_selection_binding(
    receipt: Any,
    selection: ProductionFoundationSelection,
    requests: Sequence[NativeShotRequest],
    *,
    reference_manifest: str | Path | None = None,
) -> None:
    """Bind every production shot receipt to the exact selected request and runtime.

    Quality-first production acceptance must not depend on aggregate receipt fields
    alone. Every shot must independently prove the selected profile/origin, exact
    pinned foundation provenance, requested scene/shot identity and request hash,
    plus the full CUDA execution policy chosen from the live quality-first preflight.

    When production conditioning provenance is present, this validator also reopens
    the immutable approved-reference manifest and binds both its exact digest and each
    consumed reference digest to the per-shot runtime evidence. This prevents a
    substituted image from being accepted merely because it retained an approved ID.

    This is intentionally stricter than generic/legacy connected-receipt handling:
    the quality-first production entrypoint only accepts real per-shot evidence and
    fails closed if it is absent, truncated, reordered, replayed, or rendered with a
    different memory/offload policy than the selected production plan.
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
    expected_plan = selection.plan
    expected_policy_fields = {
        "memory_strategy": expected_plan.memory_strategy,
        "enable_vae_tiling": expected_plan.enable_vae_tiling,
        "enable_vae_slicing": expected_plan.enable_vae_slicing,
        "enable_attention_slicing": expected_plan.enable_attention_slicing,
        "estimated_model_vram_gb": expected_plan.estimated_model_vram_gb,
    }
    reference_loader: ProductionReferenceLoader | None = None

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

        expected_hashes: tuple[str, ...] | None = None
        if (
            hasattr(result, "conditioning_provenance")
            and request.approved_reference_ids
        ):
            if reference_manifest is None:
                raise GPUProductionBenchmarkCLIError(
                    f"shot {index} has production conditioning evidence without an approved reference manifest"
                )
            try:
                if reference_loader is None:
                    reference_loader = ProductionReferenceLoader(reference_manifest)
                expected_hashes = tuple(
                    reference_loader.reference_sha256(reference_id)
                    for reference_id in request.approved_reference_ids
                )
            except ProductionReferenceError as exc:
                raise GPUProductionBenchmarkCLIError(str(exc)) from exc

        _validate_conditioning_binding(
            result,
            request,
            shot_index=index,
            expected_reference_hashes=expected_hashes,
        )

        execution_plan = getattr(shot_receipt, "execution_plan", None)
        if execution_plan is None:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} is missing GPU execution-plan evidence"
            )
        if getattr(execution_plan, "device", None) != expected_plan.device:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} CUDA device does not match quality-first selection"
            )
        if getattr(execution_plan, "dtype", None) != expected_plan.dtype:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} dtype does not match quality-first selection"
            )
        for field, expected_value in expected_policy_fields.items():
            if getattr(execution_plan, field, None) != expected_value:
                raise GPUProductionBenchmarkCLIError(
                    f"shot {index} GPU execution field {field!r} does not match "
                    "quality-first selection"
                )

        runtime = getattr(shot_receipt, "runtime_provenance", None)
        if not isinstance(runtime, dict):
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} is missing runtime provenance"
            )
        if runtime.get("cuda_device") != expected_plan.device:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} runtime CUDA device does not match quality-first selection"
            )
        if runtime.get("dtype") != expected_plan.dtype:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} runtime dtype does not match quality-first selection"
            )
        if expected_hashes is not None and reference_loader is not None:
            reference_assets = runtime.get("reference_assets")
            if not isinstance(reference_assets, dict):
                raise GPUProductionBenchmarkCLIError(
                    f"shot {index} runtime is missing approved reference asset provenance"
                )
            if (
                reference_assets.get("manifest_sha256")
                != reference_loader.manifest_sha256
            ):
                raise GPUProductionBenchmarkCLIError(
                    f"shot {index} runtime reference manifest does not match approved manifest"
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
    try:
        remove_stale_quality_first_attestation(output_root)
    except ProductionBenchmarkAttestationError as exc:
        raise GPUProductionBenchmarkCLIError(str(exc)) from exc

    quality_evaluator = _production_semantic_quality_evaluator(
        requests, reference_manifest
    )
    transition_evaluator = (
        _production_transition_evaluator(quality_evaluator)
        if isinstance(quality_evaluator, ArtifactMeasuredSequenceQualityEvaluator)
        else None
    )

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
    _validate_per_shot_selection_binding(
        receipt,
        selection,
        requests,
        reference_manifest=reference_manifest,
    )
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

    try:
        write_quality_first_production_attestation(
            output_root,
            selection_manifest=selection_manifest,
            receipt=receipt,
        )
    except ProductionBenchmarkAttestationError as exc:
        raise GPUProductionBenchmarkCLIError(str(exc)) from exc
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
