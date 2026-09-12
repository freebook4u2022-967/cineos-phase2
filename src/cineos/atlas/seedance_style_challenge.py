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


def _conditioned_character_ids(request: NativeShotRequest) -> tuple[str, ...]:
    """Return distinct conditioned cast identities without inventing identity aliases."""

    identities: list[str] = []
    seen: set[str] = set()
    for index, character in enumerate(request.characters):
        if not isinstance(character, Mapping):
            raise SeedanceStyleChallengeError(
                f"shot {request.scene_id}/{request.shot_id} characters[{index}] must be a mapping"
            )
        canonical = character.get("character_uuid")
        legacy = character.get("character_id")
        if canonical is not None and (
            not isinstance(canonical, str) or not canonical.strip()
        ):
            raise SeedanceStyleChallengeError(
                f"shot {request.scene_id}/{request.shot_id} characters[{index}].character_uuid "
                "must be non-empty when supplied"
            )
        if legacy is not None and (not isinstance(legacy, str) or not legacy.strip()):
            raise SeedanceStyleChallengeError(
                f"shot {request.scene_id}/{request.shot_id} characters[{index}].character_id "
                "must be non-empty when supplied"
            )
        if canonical is not None and legacy is not None:
            if canonical.strip() != legacy.strip():
                raise SeedanceStyleChallengeError(
                    f"shot {request.scene_id}/{request.shot_id} characters[{index}] has "
                    "conflicting character_uuid/character_id"
                )
            identity = canonical.strip()
        elif canonical is not None:
            identity = canonical.strip()
        elif legacy is not None:
            identity = legacy.strip()
        else:
            continue
        if identity not in seen:
            identities.append(identity)
            seen.add(identity)
    return tuple(identities)


def _validate_structural_challenge_grounding(
    request: NativeShotRequest,
    challenges: Sequence[str],
) -> None:
    """Reject challenge labels that contradict directly observable request structure.

    This intentionally enforces only hard cases whose prerequisites are unambiguous
    in the native request contract. It does not pretend that structural declarations
    prove visual quality; artifact-bound GPU/QC evidence remains the quality authority.
    """

    challenge_set = set(challenges)
    shot_key = f"{request.scene_id}/{request.shot_id}"
    if "multi_character_interaction" in challenge_set:
        character_ids = _conditioned_character_ids(request)
        if len(character_ids) < 2:
            raise SeedanceStyleChallengeError(
                f"shot {shot_key} declares multi_character_interaction but has fewer "
                "than two distinct conditioned character identities"
            )
    if "object_interaction" in challenge_set:
        if not isinstance(request.props, list) or not request.props:
            raise SeedanceStyleChallengeError(
                f"shot {shot_key} declares object_interaction but has no declared props"
            )
        for index, prop in enumerate(request.props):
            if not isinstance(prop, Mapping) or not prop:
                raise SeedanceStyleChallengeError(
                    f"shot {shot_key} props[{index}] must be a non-empty mapping for "
                    "object_interaction"
                )


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
            "schema": "cineos-seedance-style-challenge-coverage/0.2",
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


@dataclass(frozen=True, slots=True)
class ChallengeBoundGPUConnectedBenchmarkReceipt(GPUConnectedBenchmarkReceipt):
    """Connected receipt whose serialized evidence includes benchmark hard-case scope.

    This remains an instance of :class:`GPUConnectedBenchmarkReceipt` so existing
    consumers keep their attribute/type contract. Only competitive benchmark paths
    construct it; generic and legacy connected benchmarks remain unchanged.
    """

    competitive_challenge_contract: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = super(ChallengeBoundGPUConnectedBenchmarkReceipt, self).to_dict()
        payload["competitive_challenge_contract"] = dict(
            self.competitive_challenge_contract or {}
        )
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
        challenges = _normalized_challenges(request)
        _validate_structural_challenge_grounding(request, challenges)
        for challenge in challenges:
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


def bind_challenge_coverage(
    receipt: GPUConnectedBenchmarkReceipt,
    coverage: ChallengeCoverage,
) -> ChallengeBoundGPUConnectedBenchmarkReceipt:
    """Bind declared difficult-case scope to both receipt serialization and manifest."""

    contract = coverage.to_dict()
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

    bound = ChallengeBoundGPUConnectedBenchmarkReceipt(
        benchmark_id=receipt.benchmark_id,
        profile_id=receipt.profile_id,
        origin=receipt.origin,
        shot_receipts=receipt.shot_receipts,
        chain_sha256=receipt.chain_sha256,
        total_output_bytes=receipt.total_output_bytes,
        elapsed_seconds=receipt.elapsed_seconds,
        manifest_path=receipt.manifest_path,
        quality_reports=receipt.quality_reports,
        dialogue_shot_ids=receipt.dialogue_shot_ids,
        competitive_challenge_contract=contract,
    )
    payload["competitive_challenge_contract"] = contract
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
    return bound


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
    return bind_challenge_coverage(receipt, coverage)


__all__ = [
    "CHALLENGE_METADATA_KEY",
    "REQUIRED_CHALLENGES",
    "ChallengeBoundGPUConnectedBenchmarkReceipt",
    "ChallengeCoverage",
    "SeedanceStyleChallengeError",
    "bind_challenge_coverage",
    "run_seedance_style_gpu_benchmark",
    "validate_challenge_coverage",
]
