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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .gpu_benchmark_cli import (
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

    result = validate_connected_benchmark_fixture(fixture_path, shot_count=len(requests))
    ordered_shot_ids, bundle_sha256 = _normalized_request_bundle_binding(requests)
    result.update(
        {
            "schema": "cineos-connected-benchmark-fixture-preflight/0.2",
            "request_manifest_path": Path(requests_path).as_posix(),
            "ordered_shot_ids": list(ordered_shot_ids),
            "normalized_request_bundle_sha256": bundle_sha256,
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
