"""Fail-closed challenge contract for CINEOS competitive connected-film benchmarks.

This module does not claim parity with Seedance or any other external system. It
only guarantees that a CINEOS benchmark sequence declares coverage of the hard
production cases that must be exercised before comparative quality claims are
considered. Actual quality still comes from artifact-bound GPU/QC evidence.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .foundation_profiles import FoundationExecutionProfile
from .gpu_connected_benchmark import (
    GPUConnectedBenchmarkError,
    GPUConnectedBenchmarkReceipt,
    run_connected_gpu_benchmark,
)
from .native_request import NativeShotRequest

CHALLENGE_METADATA_KEY = "benchmark_challenges"
REQUIRED_CHALLENGES = (
    "identity_consistency",
    "multi_character_interaction",
    "hands_anatomy",
    "locomotion",
    "dialogue",
    "object_interaction",
    "fast_camera_movement",
    "lighting_change",
    "physics",
)


class SeedanceStyleChallengeError(GPUConnectedBenchmarkError):
    """Raised when a competitive benchmark plan is incomplete or ambiguous."""


def _normalized_challenges(request: NativeShotRequest) -> tuple[str, ...]:
    raw = request.metadata.get(CHALLENGE_METADATA_KEY)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise SeedanceStyleChallengeError(
            f"shot {request.scene_id}/{request.shot_id} must declare "
            f"metadata[{CHALLENGE_METADATA_KEY!r}] as a sequence"
        )

    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            raise SeedanceStyleChallengeError(
                f"shot {request.scene_id}/{request.shot_id} has an invalid benchmark challenge"
            )
        challenge = value.strip().lower()
        if challenge not in REQUIRED_CHALLENGES:
            raise SeedanceStyleChallengeError(
                f"shot {request.scene_id}/{request.shot_id} declares unsupported "
                f"benchmark challenge {challenge!r}"
            )
        if challenge not in seen:
            normalized.append(challenge)
            seen.add(challenge)
    if not normalized:
        raise SeedanceStyleChallengeError(
            f"shot {request.scene_id}/{request.shot_id} declares no benchmark challenges"
        )
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class ChallengeCoverage:
    """Auditable declaration of which hard cases are exercised by which shots."""

    challenge_to_shots: Mapping[str, tuple[str, ...]]

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(
            challenge
            for challenge in REQUIRED_CHALLENGES
            if not self.challenge_to_shots.get(challenge)
        )

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "cineos-seedance-style-challenge-coverage/0.1",
            "required_challenges": list(REQUIRED_CHALLENGES),
            "complete": self.complete,
            "missing": list(self.missing),
            "challenge_to_shots": {
                challenge: list(self.challenge_to_shots.get(challenge, ()))
                for challenge in REQUIRED_CHALLENGES
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["contract_sha256"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
        return payload


def validate_challenge_coverage(
    requests: Sequence[NativeShotRequest],
) -> ChallengeCoverage:
    """Require all agreed difficult cases across a connected 5-10-shot sequence."""

    if not 5 <= len(requests) <= 10:
        raise SeedanceStyleChallengeError(
            "competitive challenge coverage requires between 5 and 10 shots"
        )

    coverage: dict[str, list[str]] = {
        challenge: [] for challenge in REQUIRED_CHALLENGES
    }
    for request in requests:
        shot_key = f"{request.scene_id}/{request.shot_id}"
        for challenge in _normalized_challenges(request):
            coverage[challenge].append(shot_key)

    frozen = ChallengeCoverage(
        challenge_to_shots={
            challenge: tuple(shots) for challenge, shots in coverage.items()
        }
    )
    if frozen.missing:
        raise SeedanceStyleChallengeError(
            "competitive benchmark is missing required challenge coverage: "
            + ", ".join(frozen.missing)
        )
    return frozen


def _bind_coverage_to_manifest(
    receipt: GPUConnectedBenchmarkReceipt,
    coverage: ChallengeCoverage,
) -> None:
    manifest = Path(receipt.manifest_path)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SeedanceStyleChallengeError(
            f"cannot read connected benchmark manifest for challenge binding: {manifest}"
        ) from exc
    if not isinstance(payload, dict):
        raise SeedanceStyleChallengeError(
            "connected benchmark manifest must be a JSON object"
        )
    if payload.get("chain_sha256") != receipt.chain_sha256:
        raise SeedanceStyleChallengeError(
            "connected benchmark manifest chain hash does not match completed receipt"
        )

    payload["competitive_challenge_contract"] = coverage.to_dict()
    temporary = manifest.with_suffix(manifest.suffix + ".challenge.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise SeedanceStyleChallengeError(
            f"cannot bind competitive challenge contract to benchmark manifest: {manifest}"
        ) from exc


def run_seedance_style_gpu_benchmark(
    benchmark_id: str,
    requests: Sequence[NativeShotRequest],
    profile: FoundationExecutionProfile,
    **kwargs: Any,
) -> GPUConnectedBenchmarkReceipt:
    """Run the existing connected benchmark only after hard-case coverage is complete.

    The wrapper deliberately leaves GPU execution, provenance, artifact validation,
    and measured QC ownership in ``run_connected_gpu_benchmark``. It adds only an
    auditable challenge-coverage contract and therefore cannot turn declared test
    intent into quality evidence.
    """

    coverage = validate_challenge_coverage(requests)
    receipt = run_connected_gpu_benchmark(
        benchmark_id,
        requests,
        profile,
        **kwargs,
    )
    _bind_coverage_to_manifest(receipt, coverage)
    return receipt


__all__ = [
    "CHALLENGE_METADATA_KEY",
    "REQUIRED_CHALLENGES",
    "ChallengeCoverage",
    "SeedanceStyleChallengeError",
    "run_seedance_style_gpu_benchmark",
    "validate_challenge_coverage",
]
