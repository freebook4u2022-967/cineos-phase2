"""Fail-fast validation for real connected-production GPU inputs.

This module intentionally runs before heavyweight foundation/QC model acquisition on
self-hosted GPU workers. It reuses the same connected-shot and production-reference
contracts as the execution path so malformed continuity graphs or stale approved
identity assets cannot consume scarce model-download/load time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .gpu_benchmark_cli import (
    COMPETITIVE_CHALLENGE_METADATA_KEY,
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


_OBJECT_INTERACTION_ACTION_TERMS = frozenset(
    {
        "touch",
        "touching",
        "hold",
        "holding",
        "grasp",
        "grasping",
        "grab",
        "grabbing",
        "grip",
        "gripping",
        "reach",
        "reaching",
        "pick",
        "picking",
        "pick_up",
        "pickup",
        "carry",
        "carrying",
        "hand",
        "handing",
        "pass",
        "passing",
        "give",
        "giving",
        "receive",
        "receiving",
        "open",
        "opening",
        "close",
        "closing",
        "throw",
        "throwing",
        "catch",
        "catching",
        "drop",
        "dropping",
        "push",
        "pushing",
        "pull",
        "pulling",
        "place",
        "placing",
        "put",
        "putting",
        "use",
        "using",
    }
)
_PROP_ID_KEYS = ("prop_uuid", "prop_id", "object_id", "id")


def _request_bundle_sha256(requests: Sequence[NativeShotRequest]) -> str:
    """Hash the exact ordered request bundle accepted by production preflight."""

    payload = json.dumps(
        [request.to_dict() for request in requests],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_aliased_reference_content(reference_hashes: dict[str, str]) -> None:
    """Reject distinct approved IDs that resolve to identical identity content.

    A production multi-reference run must not be able to claim multiple approved
    identities by assigning different reference IDs to the same underlying asset.
    This check intentionally happens before image decode or heavyweight model IO.
    """

    ids_by_hash: dict[str, list[str]] = {}
    for reference_id, digest in reference_hashes.items():
        ids_by_hash.setdefault(digest, []).append(reference_id)
    aliases = [ids for ids in ids_by_hash.values() if len(ids) > 1]
    if not aliases:
        return

    alias_text = "; ".join(", ".join(sorted(ids)) for ids in aliases)
    raise ProductionInputPreflightError(
        "production reference IDs must resolve to distinct approved content; "
        f"duplicate-content aliases: {alias_text}"
    )


def _reject_stale_request_hashes(requests: Sequence[NativeShotRequest]) -> None:
    """Reject requests mutated after their native content identity was established."""

    for request in requests:
        if request.content_hash_is_current():
            continue
        raise ProductionInputPreflightError(
            f"shot {request.scene_id}/{request.shot_id} has a stale or missing "
            "content_hash; call refresh_hash() after mutating NativeShotRequest"
        )


def _normalized_terms(value: Any) -> set[str]:
    """Extract conservative normalized terms from structured native conditioning."""

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


def _prop_identity(prop: Mapping[str, Any]) -> str | None:
    """Return one stable conditioned prop identity while supporting legacy aliases."""

    identities = [
        value.strip()
        for key in _PROP_ID_KEYS
        if isinstance((value := prop.get(key)), str) and value.strip()
    ]
    if not identities:
        return None
    if len(set(identities)) != 1:
        raise ProductionInputPreflightError(
            "object-interaction prop has conflicting prop identity aliases"
        )
    return identities[0]


def _cue_prop_identity(cue: Mapping[str, Any]) -> str | None:
    values = [
        value.strip()
        for key in ("prop_uuid", "prop_id", "object_id")
        if isinstance((value := cue.get(key)), str) and value.strip()
    ]
    if not values:
        return None
    if len(set(values)) != 1:
        raise ProductionInputPreflightError(
            "object_interaction_cue has conflicting prop identity aliases"
        )
    return values[0]


def _has_explicit_object_interaction_cue(
    request: NativeShotRequest,
    *,
    prop_ids: set[str],
) -> bool:
    cues = request.performance.get("object_interaction_cues")
    if cues is None:
        return False
    if not isinstance(cues, Sequence) or isinstance(cues, (str, bytes)) or not cues:
        raise ProductionInputPreflightError(
            f"shot {request.shot_id!r} object_interaction_cues must be a non-empty sequence"
        )

    conditioned_character_ids = {
        identity.strip()
        for character in request.characters
        if isinstance(character, Mapping)
        for key in ("character_uuid", "character_id")
        if isinstance((identity := character.get(key)), str) and identity.strip()
    }
    found = False
    for cue_index, cue in enumerate(cues):
        if not isinstance(cue, Mapping):
            raise ProductionInputPreflightError(
                f"shot {request.shot_id!r} object interaction cue {cue_index} must be an object"
            )
        prop_id = _cue_prop_identity(cue)
        if prop_id is None or prop_id not in prop_ids:
            raise ProductionInputPreflightError(
                f"shot {request.shot_id!r} object interaction cue {cue_index} must reference "
                "a conditioned prop identity"
            )
        character_id = cue.get("character_id", cue.get("character_uuid"))
        if character_id is not None:
            if (
                not isinstance(character_id, str)
                or character_id.strip() not in conditioned_character_ids
            ):
                raise ProductionInputPreflightError(
                    f"shot {request.shot_id!r} object interaction cue {cue_index} references "
                    "an unconditioned character identity"
                )
        cue_terms = _normalized_terms((cue.get("action"), cue.get("description")))
        if not cue_terms & _OBJECT_INTERACTION_ACTION_TERMS:
            raise ProductionInputPreflightError(
                f"shot {request.shot_id!r} object interaction cue {cue_index} contains no "
                "explicit manipulation action"
            )
        found = True
    return found


def _validate_object_interaction_grounding(
    requests: Sequence[NativeShotRequest],
) -> None:
    """Bind object-interaction benchmark claims to real prop manipulation conditioning.

    A prop merely being present in the request is not evidence that a difficult
    character-object interaction was requested. For every shot carrying the benchmark
    tag, at least one identifiable conditioned prop must either be named in an explicit
    manipulation performance instruction or be referenced by a structured
    ``object_interaction_cues`` entry.
    """

    for request in requests:
        raw_tags = request.metadata.get(COMPETITIVE_CHALLENGE_METADATA_KEY, ())
        if not isinstance(raw_tags, Sequence) or isinstance(raw_tags, (str, bytes)):
            continue
        if "object_interaction" not in raw_tags:
            continue

        prop_ids: set[str] = set()
        for prop in request.props:
            if not isinstance(prop, Mapping):
                raise ProductionInputPreflightError(
                    f"shot {request.shot_id!r} object_interaction requires structured prop conditioning"
                )
            prop_id = _prop_identity(prop)
            if prop_id is None:
                raise ProductionInputPreflightError(
                    f"shot {request.shot_id!r} object_interaction prop requires a stable prop identity"
                )
            prop_ids.add(prop_id)

        if not prop_ids:
            raise ProductionInputPreflightError(
                f"shot {request.shot_id!r} object_interaction contains no identifiable conditioned prop"
            )
        if _has_explicit_object_interaction_cue(request, prop_ids=prop_ids):
            continue

        performance_terms = _normalized_terms(
            (
                request.performance.get("action"),
                request.performance.get("gesture_tracks", ()),
                request.performance.get("body_performance_tracks", ()),
            )
        )
        has_manipulation = bool(performance_terms & _OBJECT_INTERACTION_ACTION_TERMS)
        referenced_prop_ids = {
            prop_id
            for prop_id in prop_ids
            if prop_id.strip().lower().replace("-", "_") in performance_terms
        }
        if not has_manipulation or not referenced_prop_ids:
            raise ProductionInputPreflightError(
                f"shot {request.shot_id!r} declares object_interaction but contains no explicit "
                "performance manipulation grounded to a conditioned prop identity"
            )


def preflight_production_inputs(
    requests: Sequence[NativeShotRequest],
    reference_manifest: str | Path,
) -> dict[str, object]:
    """Validate connected production inputs without loading models."""

    request_sequence = tuple(requests)
    _reject_stale_request_hashes(request_sequence)
    _validate_object_interaction_grounding(request_sequence)
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
        reference_hashes = {
            reference_id: loader.reference_sha256(reference_id)
            for reference_id in requested_reference_ids
        }
        _reject_aliased_reference_content(reference_hashes)

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
        "schema": "cineos-production-input-preflight/0.5",
        "shot_count": len(request_sequence),
        "request_bundle_sha256": _request_bundle_sha256(request_sequence),
        "request_content_hashes": [
            request.content_hash for request in request_sequence
        ],
        "reference_count": len(requested_reference_ids),
        "distinct_reference_content_count": len(set(reference_hashes.values())),
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
