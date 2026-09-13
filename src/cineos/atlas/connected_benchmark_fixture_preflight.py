"""Bind the canonical Seedance-style connected-film fixture to real GPU requests.

The competitive suite is descriptive metadata until its connected-film fixture is
proven to describe the exact executable request bundle. This preflight deliberately
reuses :func:`load_native_requests`, which owns the concrete per-shot challenge
semantics, rather than maintaining a second and eventually divergent validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .gpu_benchmark_cli import (
    COMPETITIVE_CHALLENGE_METADATA_KEY,
    REQUIRED_COMPETITIVE_CHALLENGES,
    GPUProductionBenchmarkCLIError,
    load_native_requests,
)
from .native_request import NativeShotRequest


class ConnectedBenchmarkFixturePreflightError(RuntimeError):
    """Raised when benchmark metadata is not bound to executable production inputs."""


_CONNECTED_FILM_CASE_ID = "competitive-connected-film"
_MIN_CONNECTED_SHOTS = 5
_MAX_CONNECTED_SHOTS = 10
_REQUIRED_FILM_CAPABILITIES = (
    "identity_lock",
    "scene_memory",
    "automatic_qc",
    "audio",
    "film_assembly",
)


def _fixture_bytes(path: str | Path) -> tuple[Path, bytes]:
    fixture_path = Path(path)
    try:
        payload = fixture_path.read_bytes()
    except OSError as exc:
        raise ConnectedBenchmarkFixturePreflightError(
            f"unable to read connected benchmark fixture {fixture_path}: {exc}"
        ) from exc
    return fixture_path, payload


def _load_fixture(path: str | Path) -> tuple[Path, bytes, Mapping[str, Any]]:
    fixture_path, raw = _fixture_bytes(path)
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConnectedBenchmarkFixturePreflightError(
            f"connected benchmark fixture is not valid UTF-8 JSON: {fixture_path}"
        ) from exc
    if not isinstance(decoded, Mapping):
        raise ConnectedBenchmarkFixturePreflightError(
            "connected benchmark fixture must contain a JSON object"
        )
    return fixture_path, raw, decoded


def _string_sequence(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConnectedBenchmarkFixturePreflightError(
            f"connected benchmark fixture {field} must be a sequence"
        )
    normalized = tuple(item for item in value if isinstance(item, str) and item)
    if len(normalized) != len(value) or len(set(normalized)) != len(normalized):
        raise ConnectedBenchmarkFixturePreflightError(
            f"connected benchmark fixture {field} must contain unique non-empty strings"
        )
    return normalized


def _normalized_request_bundle_binding(
    requests: Sequence[NativeShotRequest],
) -> tuple[tuple[str, ...], str]:
    """Hash-bind the exact ordered validated request bundle used for GPU execution.

    Each request ``content_hash`` is already recomputed and verified by
    :func:`load_native_requests`. Hashing the ordered ``shot_id``/``content_hash`` pairs
    therefore binds both request semantics and sequence order while remaining stable
    across irrelevant JSON whitespace or object-key formatting changes.
    """

    ordered_shot_ids: list[str] = []
    entries: list[dict[str, str]] = []
    for index, request in enumerate(requests):
        shot_id = request.shot_id
        content_hash = request.content_hash
        if not isinstance(shot_id, str) or not shot_id:
            raise ConnectedBenchmarkFixturePreflightError(
                f"validated request {index} is missing shot_id"
            )
        if (
            not isinstance(content_hash, str)
            or len(content_hash) != 64
            or any(character not in "0123456789abcdef" for character in content_hash)
        ):
            raise ConnectedBenchmarkFixturePreflightError(
                f"validated request {index} is missing a canonical SHA-256 content_hash"
            )
        ordered_shot_ids.append(shot_id)
        entries.append({"shot_id": shot_id, "content_hash": content_hash})

    canonical = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return tuple(ordered_shot_ids), hashlib.sha256(canonical).hexdigest()


def _challenge_shot_bindings(
    requests: Sequence[NativeShotRequest],
) -> dict[str, list[str]]:
    """Bind every mandatory difficult-case claim to the exact validated shot IDs.

    ``load_native_requests`` already verifies the semantics behind each challenge tag.
    Persisting the resulting challenge-to-shot mapping here prevents downstream GPU or
    delivery evidence from presenting only an aggregate coverage list with no auditable
    link back to the requests that actually exercised each stressor.
    """

    bindings = {challenge: [] for challenge in sorted(REQUIRED_COMPETITIVE_CHALLENGES)}
    for index, request in enumerate(requests):
        metadata = getattr(request, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise ConnectedBenchmarkFixturePreflightError(
                f"validated request {index} is missing benchmark metadata"
            )
        raw_tags = metadata.get(COMPETITIVE_CHALLENGE_METADATA_KEY)
        if not isinstance(raw_tags, Sequence) or isinstance(raw_tags, (str, bytes)):
            raise ConnectedBenchmarkFixturePreflightError(
                f"validated request {index} is missing competitive challenge tags"
            )
        shot_id = getattr(request, "shot_id", None)
        if not isinstance(shot_id, str) or not shot_id:
            raise ConnectedBenchmarkFixturePreflightError(
                f"validated request {index} is missing shot_id"
            )
        for challenge in raw_tags:
            if challenge not in REQUIRED_COMPETITIVE_CHALLENGES:
                raise ConnectedBenchmarkFixturePreflightError(
                    f"validated request {index} contains unknown competitive challenge "
                    f"{challenge!r}"
                )
            if shot_id not in bindings[challenge]:
                bindings[challenge].append(shot_id)

    missing = [challenge for challenge, shot_ids in bindings.items() if not shot_ids]
    if missing:
        raise ConnectedBenchmarkFixturePreflightError(
            "validated request bundle lost mandatory competitive challenge coverage: "
            + ", ".join(missing)
        )
    return bindings


def _validate_multi_character_identity_assignment(
    requests: Sequence[NativeShotRequest],
) -> tuple[str, ...]:
    """Require explicit per-character identity sources for competitive cast shots.

    Shot-level approved references establish that assets are allowed for production,
    but cardinality alone cannot prove which source belongs to which character.  A
    two-character shot with two unrelated approved images must not be accepted merely
    because the counts happen to match.  Native request validation already rejects
    unapproved and cross-character duplicated local references; this benchmark gate
    additionally requires complete local ownership for every multi-character shot.
    """

    bound_shot_ids: list[str] = []
    for request_index, request in enumerate(requests):
        characters = getattr(request, "characters", None)
        if not isinstance(characters, list) or len(characters) < 2:
            continue

        shot_id = getattr(request, "shot_id", f"request-{request_index}")
        for character_index, character in enumerate(characters):
            if not isinstance(character, dict):
                raise ConnectedBenchmarkFixturePreflightError(
                    f"multi-character shot {shot_id!r} has invalid characters[{character_index}]"
                )
            reference_ids = character.get("approved_reference_ids")
            if not isinstance(reference_ids, (list, tuple)) or not reference_ids:
                raise ConnectedBenchmarkFixturePreflightError(
                    "multi-character competitive benchmark requires explicit "
                    "character-local approved_reference_ids for every cast member; "
                    f"shot {shot_id!r} characters[{character_index}] is unbound"
                )
        bound_shot_ids.append(str(shot_id))

    return tuple(bound_shot_ids)


def _character_identity(request_character: Mapping[str, Any], *, field: str) -> str:
    """Resolve a character identity without silently accepting conflicting aliases."""

    canonical = request_character.get("character_uuid")
    legacy = request_character.get("character_id")
    for name, value in (("character_uuid", canonical), ("character_id", legacy)):
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConnectedBenchmarkFixturePreflightError(
                f"{field}.{name} must be a non-empty string when supplied"
            )
    if canonical is not None and legacy is not None:
        if canonical.strip() != legacy.strip():
            raise ConnectedBenchmarkFixturePreflightError(
                f"{field} has conflicting character_uuid/character_id"
            )
        return canonical.strip()
    identity = canonical if canonical is not None else legacy
    if identity is None:
        raise ConnectedBenchmarkFixturePreflightError(
            f"{field} requires character_uuid or character_id"
        )
    return identity.strip()


def _identity_reference_bindings(
    request: NativeShotRequest, *, request_index: int
) -> set[tuple[str, str]]:
    """Return explicit character-to-reference bindings for one validated shot.

    Multi-character production shots must have local reference ownership before this
    function runs. For a single-character legacy request, shot-level approved references
    remain safely attributable to the sole conditioned identity for compatibility.
    """

    characters = getattr(request, "characters", None)
    if not isinstance(characters, list) or not characters:
        return set()

    shot_approved = set(request.approved_reference_ids)
    bindings: set[tuple[str, str]] = set()
    for character_index, character in enumerate(characters):
        if not isinstance(character, Mapping):
            raise ConnectedBenchmarkFixturePreflightError(
                f"shot {request.shot_id!r} has invalid characters[{character_index}]"
            )
        identity = _character_identity(
            character,
            field=f"shot {request.shot_id!r} characters[{character_index}]",
        )
        local_references = character.get("approved_reference_ids")
        if local_references is None:
            if len(characters) == 1:
                local_references = tuple(request.approved_reference_ids)
            else:
                continue
        if not isinstance(local_references, Sequence) or isinstance(
            local_references, (str, bytes)
        ):
            raise ConnectedBenchmarkFixturePreflightError(
                f"shot {request.shot_id!r} character {identity!r} approved_reference_ids "
                "must be a sequence"
            )
        for reference_id in local_references:
            if not isinstance(reference_id, str) or not reference_id:
                raise ConnectedBenchmarkFixturePreflightError(
                    f"shot {request.shot_id!r} character {identity!r} contains an "
                    "invalid approved_reference_id"
                )
            if reference_id not in shot_approved:
                raise ConnectedBenchmarkFixturePreflightError(
                    f"shot {request.shot_id!r} character {identity!r} binds unapproved "
                    f"reference {reference_id!r}"
                )
            bindings.add((identity, reference_id))
    return bindings


def _validate_identity_consistency_bindings(
    requests: Sequence[NativeShotRequest],
) -> dict[str, list[dict[str, str]]]:
    """Prove identity consistency with stable character/reference ownership.

    Repeating a reference token is insufficient: the same reference could be attached
    to a different character on a later shot. The competitive identity challenge now
    requires every identity-tagged shot to contain at least one ``(character, reference)``
    pair that also occurs in another connected shot. The returned evidence is explicit
    and auditable rather than an aggregate boolean.
    """

    bindings_by_shot: list[set[tuple[str, str]]] = [
        _identity_reference_bindings(request, request_index=index)
        for index, request in enumerate(requests)
    ]
    occurrences = Counter(
        binding for shot_bindings in bindings_by_shot for binding in shot_bindings
    )
    evidence: dict[str, list[dict[str, str]]] = {}

    for index, request in enumerate(requests):
        raw_tags = request.metadata.get(COMPETITIVE_CHALLENGE_METADATA_KEY, ())
        if "identity_consistency" not in raw_tags:
            continue
        persistent = sorted(
            binding for binding in bindings_by_shot[index] if occurrences[binding] >= 2
        )
        if not persistent:
            raise ConnectedBenchmarkFixturePreflightError(
                f"shot {request.shot_id!r} declares identity_consistency but no "
                "character/reference identity binding persists into another connected shot"
            )
        evidence[request.shot_id] = [
            {"character_id": identity, "approved_reference_id": reference_id}
            for identity, reference_id in persistent
        ]

    if not evidence:
        raise ConnectedBenchmarkFixturePreflightError(
            "connected benchmark contains no identity_consistency challenge evidence"
        )
    return evidence


def validate_connected_benchmark_fixture(
    fixture_path: str | Path,
    *,
    shot_count: int,
) -> dict[str, object]:
    """Validate and hash-bind the canonical connected-film benchmark declaration."""

    path, raw, payload = _load_fixture(fixture_path)
    if payload.get("case_id") != _CONNECTED_FILM_CASE_ID:
        raise ConnectedBenchmarkFixturePreflightError(
            f"connected benchmark fixture case_id must be {_CONNECTED_FILM_CASE_ID!r}"
        )
    if payload.get("deterministic") is not True:
        raise ConnectedBenchmarkFixturePreflightError(
            "connected benchmark fixture must require deterministic execution"
        )
    if payload.get("real_inference_required") is not True:
        raise ConnectedBenchmarkFixturePreflightError(
            "connected benchmark fixture must require real inference"
        )
    if payload.get("minimum_connected_shots") != _MIN_CONNECTED_SHOTS:
        raise ConnectedBenchmarkFixturePreflightError(
            "connected benchmark fixture minimum_connected_shots must be 5"
        )
    if payload.get("maximum_connected_shots") != _MAX_CONNECTED_SHOTS:
        raise ConnectedBenchmarkFixturePreflightError(
            "connected benchmark fixture maximum_connected_shots must be 10"
        )
    if not _MIN_CONNECTED_SHOTS <= shot_count <= _MAX_CONNECTED_SHOTS:
        raise ConnectedBenchmarkFixturePreflightError(
            f"connected benchmark request bundle requires 5-10 shots; received {shot_count}"
        )

    film_capabilities = _string_sequence(
        payload.get("benchmark_challenges"), field="benchmark_challenges"
    )
    if film_capabilities != _REQUIRED_FILM_CAPABILITIES:
        raise ConnectedBenchmarkFixturePreflightError(
            "connected benchmark fixture film capabilities do not match the canonical "
            "complete-film contract"
        )

    challenge_tags = _string_sequence(
        payload.get("required_competitive_challenges"),
        field="required_competitive_challenges",
    )
    expected_challenges = tuple(sorted(REQUIRED_COMPETITIVE_CHALLENGES))
    if challenge_tags != expected_challenges:
        raise ConnectedBenchmarkFixturePreflightError(
            "connected benchmark fixture competitive challenges do not match the "
            "executable GPU benchmark contract"
        )

    return {
        "schema": "cineos-connected-benchmark-fixture-preflight/0.1",
        "case_id": _CONNECTED_FILM_CASE_ID,
        "fixture_path": path.as_posix(),
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "shot_count": shot_count,
        "required_competitive_challenges": list(expected_challenges),
        "real_inference_required": True,
        "validated": True,
    }


def preflight_connected_benchmark_fixture(
    requests_path: str | Path,
    fixture_path: str | Path,
) -> dict[str, object]:
    """Bind fixture metadata to the exact normalized requests passing GPU validation."""

    try:
        requests = load_native_requests(requests_path)
    except GPUProductionBenchmarkCLIError as exc:
        raise ConnectedBenchmarkFixturePreflightError(str(exc)) from exc

    result = validate_connected_benchmark_fixture(
        fixture_path, shot_count=len(requests)
    )
    identity_assignment_shot_ids = _validate_multi_character_identity_assignment(
        requests
    )
    identity_consistency_bindings = _validate_identity_consistency_bindings(requests)
    challenge_shot_ids = _challenge_shot_bindings(requests)
    ordered_shot_ids, bundle_sha256 = _normalized_request_bundle_binding(requests)
    result.update(
        {
            "schema": "cineos-connected-benchmark-fixture-preflight/0.4",
            "request_manifest_path": Path(requests_path).as_posix(),
            "ordered_shot_ids": list(ordered_shot_ids),
            "normalized_request_bundle_sha256": bundle_sha256,
            "identity_assignment_shot_ids": list(identity_assignment_shot_ids),
            "identity_consistency_bindings": identity_consistency_bindings,
            "competitive_challenge_shot_ids": challenge_shot_ids,
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind the canonical connected-film benchmark fixture to executable CINEOS "
            "GPU requests before model acquisition."
        )
    )
    parser.add_argument("--requests", required=True)
    parser.add_argument("--benchmark-fixture", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = preflight_connected_benchmark_fixture(
            args.requests,
            args.benchmark_fixture,
        )
    except ConnectedBenchmarkFixturePreflightError as exc:
        raise SystemExit(
            f"connected benchmark fixture preflight failed: {exc}"
        ) from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ConnectedBenchmarkFixturePreflightError",
    "main",
    "preflight_connected_benchmark_fixture",
    "validate_connected_benchmark_fixture",
]
