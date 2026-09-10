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

_LOCOMOTION_TERMS = frozenset(
    {
        "walk",
        "walking",
        "run",
        "running",
        "jog",
        "jogging",
        "sprint",
        "sprinting",
    }
)
_FAST_CAMERA_MOTION_TERMS = frozenset(
    {
        "fast",
        "rapid",
        "aggressive",
        "whip",
        "whip_pan",
        "snap_pan",
        "crash_zoom",
        "speed_ramp",
        "high_speed",
    }
)
_HAND_ACTION_TERMS = frozenset(
    {
        "hand",
        "hands",
        "finger",
        "fingers",
        "grasp",
        "grasping",
        "grab",
        "grabbing",
        "grip",
        "gripping",
        "reach",
        "reaching",
        "point",
        "pointing",
        "hold",
        "holding",
    }
)
_LIGHTING_CHANGE_TERMS = frozenset(
    {
        "change",
        "changing",
        "transition",
        "transitioning",
        "flicker",
        "flickering",
        "strobe",
        "blackout",
        "sunrise",
        "sunset",
        "dawn",
        "dusk",
        "day_to_night",
        "night_to_day",
    }
)
_PHYSICS_ACTION_TERMS = frozenset(
    {
        "throw",
        "throwing",
        "catch",
        "catching",
        "drop",
        "dropping",
        "fall",
        "falling",
        "bounce",
        "bouncing",
        "collide",
        "collision",
        "impact",
        "splash",
        "splashing",
        "spill",
        "spilling",
        "break",
        "breaking",
        "roll",
        "rolling",
        "slide",
        "sliding",
        "swing",
        "swinging",
    }
)


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


def _normalized_terms(value: Any) -> set[str]:
    """Extract conservative lower-case action tokens from benchmark conditioning."""

    terms: set[str] = set()
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_")
        if normalized:
            terms.add(normalized)
            terms.update(normalized.replace("_", " ").split())
    elif isinstance(value, Mapping):
        for nested in value.values():
            terms.update(_normalized_terms(nested))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            terms.update(_normalized_terms(nested))
    return terms


def _conditioned_character_ids(request: NativeShotRequest) -> set[str]:
    """Return explicit conditioned character identities using the canonical alias rules."""

    identities: set[str] = set()
    for character in request.characters:
        if not isinstance(character, Mapping):
            continue
        canonical = character.get("character_uuid")
        legacy = character.get("character_id")
        identity = canonical if canonical not in (None, "") else legacy
        if isinstance(identity, str) and identity.strip():
            identities.add(identity.strip())
    return identities


def _validate_multi_character_interaction_conditioning(
    request: NativeShotRequest, *, index: int
) -> None:
    """Require an explicit two-plus-character interaction cue for the benchmark claim."""

    interaction_cues = request.performance.get("interaction_cues")
    if (
        not isinstance(interaction_cues, Sequence)
        or isinstance(interaction_cues, (str, bytes))
        or not interaction_cues
    ):
        raise GPUProductionBenchmarkCLIError(
            f"shot {index} declares multi_character_interaction but contains no explicit "
            "interaction_cues performance conditioning"
        )

    conditioned_ids = _conditioned_character_ids(request)
    for cue_index, cue in enumerate(interaction_cues):
        if not isinstance(cue, Mapping):
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} interaction cue {cue_index} must be an object"
            )
        participants = cue.get("participant_ids")
        if not isinstance(participants, Sequence) or isinstance(
            participants, (str, bytes)
        ):
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} interaction cue {cue_index} requires participant_ids"
            )
        participant_ids = [
            participant.strip()
            for participant in participants
            if isinstance(participant, str) and participant.strip()
        ]
        if len(participant_ids) != len(participants) or len(set(participant_ids)) < 2:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} interaction cue {cue_index} requires at least two "
                "distinct participant_ids"
            )
        unknown_ids = sorted(set(participant_ids) - conditioned_ids)
        if unknown_ids:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} interaction cue {cue_index} references unconditioned "
                f"character identities: {', '.join(unknown_ids)}"
            )
        action = cue.get("action")
        description = cue.get("description")
        if not any(
            isinstance(value, str) and value.strip() for value in (action, description)
        ):
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} interaction cue {cue_index} requires a non-empty "
                "action or description"
            )


def _has_locomotion_conditioning(request: NativeShotRequest) -> bool:
    """Return true only when native performance conditioning asks for walk/run motion."""

    action_terms = _normalized_terms(request.performance.get("action"))
    body_terms = _normalized_terms(
        request.performance.get("body_performance_tracks", [])
    )
    return bool((action_terms | body_terms) & _LOCOMOTION_TERMS)


def _has_fast_camera_motion_conditioning(request: NativeShotRequest) -> bool:
    """Require an explicit high-speed/aggressive camera cue for the camera stressor."""

    movement_terms = _normalized_terms(request.camera.get("movement"))
    return bool(movement_terms & _FAST_CAMERA_MOTION_TERMS)


def _has_hand_conditioning(request: NativeShotRequest) -> bool:
    """Require explicit hand/gesture intent for the hands-and-anatomy stressor."""

    gesture_terms = _normalized_terms(request.performance.get("gesture_tracks", []))
    action_terms = _normalized_terms(request.performance.get("action"))
    body_terms = _normalized_terms(
        request.performance.get("body_performance_tracks", [])
    )
    return bool((gesture_terms | action_terms | body_terms) & _HAND_ACTION_TERMS)


def _has_lighting_change_conditioning(request: NativeShotRequest) -> bool:
    """Require a native environment/metadata declaration of a lighting transition."""

    environment_terms = _normalized_terms(request.environment or {})
    metadata_terms = _normalized_terms(request.metadata.get("lighting_transition"))
    terms = environment_terms | metadata_terms
    if terms & _LIGHTING_CHANGE_TERMS:
        return True
    return any("_to_" in term for term in terms)


def _has_physics_conditioning(request: NativeShotRequest) -> bool:
    """Require explicit dynamic physical interaction rather than a physics label alone."""

    action_terms = _normalized_terms(request.performance.get("action"))
    body_terms = _normalized_terms(
        request.performance.get("body_performance_tracks", [])
    )
    prop_terms = _normalized_terms(request.props)
    return bool((action_terms | body_terms | prop_terms) & _PHYSICS_ACTION_TERMS)


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
            if (
                len(request.characters) < 2
                or len(set(request.approved_reference_ids)) < 2
            ):
                raise GPUProductionBenchmarkCLIError(
                    f"shot {index} declares multi_character_interaction but does not "
                    "contain at least two characters with two distinct approved "
                    "identity references"
                )
            _validate_multi_character_interaction_conditioning(request, index=index)

        if "object_interaction" in tags and not request.props:
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} declares object_interaction but contains no prop "
                "conditioning"
            )

        if "dialogue_lip_sync" in tags:
            dialogue_timing = request.performance.get("dialogue_timing")
            if (
                not isinstance(dialogue_timing, Sequence)
                or isinstance(dialogue_timing, (str, bytes))
                or not dialogue_timing
            ):
                raise GPUProductionBenchmarkCLIError(
                    f"shot {index} declares dialogue_lip_sync but contains no "
                    "dialogue_timing performance evidence"
                )

        if "walking_running" in tags and not _has_locomotion_conditioning(request):
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} declares walking_running but contains no explicit "
                "walk/run body-performance conditioning"
            )

        if "fast_camera_movement" in tags and not _has_fast_camera_motion_conditioning(
            request
        ):
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} declares fast_camera_movement but contains no explicit "
                "fast/aggressive camera-movement conditioning"
            )

        if "hands_anatomy" in tags and not _has_hand_conditioning(request):
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} declares hands_anatomy but contains no explicit "
                "hand/gesture performance conditioning"
            )

        if "lighting_changes" in tags and not _has_lighting_change_conditioning(
            request
        ):
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} declares lighting_changes but contains no explicit "
                "lighting-transition conditioning"
            )

        if "physics" in tags and not _has_physics_conditioning(request):
            raise GPUProductionBenchmarkCLIError(
                f"shot {index} declares physics but contains no explicit dynamic "
                "physical-interaction conditioning"
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
        _challenge_tags(request, index=index) for index, request in enumerate(requests)
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
