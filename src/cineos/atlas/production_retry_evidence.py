"""Fail-closed validation for production GPU reject/rerender evidence.

The connected GPU receipt contains only accepted artifacts.  Production evidence
must additionally prove how rejected attempts were corrected before those artifacts
entered the connected-film chain.  This module validates the persisted quality
retry gate against the exact accepted GPU receipts without changing renderer or
foundation provenance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class ProductionRetryEvidenceError(ValueError):
    """Raised when persisted production retry lineage is incomplete or inconsistent."""


def _non_empty(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionRetryEvidenceError(f"{field} must be a non-empty string")
    return value


def _sha256(value: Any, *, field: str) -> str:
    text = _non_empty(value, field=field)
    if len(text) != 64:
        raise ProductionRetryEvidenceError(f"{field} must be a SHA-256 digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise ProductionRetryEvidenceError(
            f"{field} must be a hexadecimal SHA-256 digest"
        ) from exc
    return text.lower()


def validate_production_quality_retry_gate(
    gate: Mapping[str, Any],
    accepted_receipts: Sequence[Any],
) -> dict[str, Any]:
    """Validate retry lineage against the exact accepted connected-shot receipts.

    The validator intentionally understands both shot-quality rejection and
    transition-quality rejection.  A non-final attempt may have passed per-shot QC
    only when the transition attempt for that same index explicitly rejected it.
    Accepted artifacts are never inferred from the manifest: their hashes and
    request hashes must match the real GPU receipts supplied by the caller.
    """

    if not isinstance(gate, Mapping):
        raise ProductionRetryEvidenceError("quality retry gate must be a mapping")
    if gate.get("schema") != "cineos-gpu-quality-retry-gate/0.2":
        raise ProductionRetryEvidenceError(
            "unsupported production quality retry schema"
        )
    if gate.get("accepted") is not True:
        raise ProductionRetryEvidenceError(
            "production quality retry gate is not accepted"
        )

    policy = gate.get("policy")
    if not isinstance(policy, Mapping):
        raise ProductionRetryEvidenceError("quality retry policy is missing")
    max_attempts = policy.get("max_attempts")
    seed_stride = policy.get("seed_stride")
    if (
        not isinstance(max_attempts, int)
        or isinstance(max_attempts, bool)
        or max_attempts < 2
    ):
        raise ProductionRetryEvidenceError("quality retry max_attempts is invalid")
    if (
        not isinstance(seed_stride, int)
        or isinstance(seed_stride, bool)
        or seed_stride <= 0
    ):
        raise ProductionRetryEvidenceError("quality retry seed_stride is invalid")

    receipts = tuple(accepted_receipts)
    shots = gate.get("shots")
    if gate.get("shot_count") != len(receipts):
        raise ProductionRetryEvidenceError(
            "quality retry shot_count does not match accepted GPU receipts"
        )
    if not isinstance(shots, list) or len(shots) != len(receipts):
        raise ProductionRetryEvidenceError(
            "quality retry shot evidence does not cover every accepted GPU receipt"
        )

    recovered_shots: list[str] = []
    total_attempts = 0
    total_rejections = 0

    for shot_position, (shot, receipt) in enumerate(zip(shots, receipts, strict=True)):
        if not isinstance(shot, Mapping):
            raise ProductionRetryEvidenceError(
                "quality retry shot entry must be a mapping"
            )
        result = getattr(receipt, "result", None)
        if result is None:
            raise ProductionRetryEvidenceError(
                "accepted GPU receipt is missing result data"
            )

        scene_id = _non_empty(shot.get("scene_id"), field="retry scene_id")
        shot_id = _non_empty(shot.get("shot_id"), field="retry shot_id")
        if scene_id != getattr(result, "scene_id", None):
            raise ProductionRetryEvidenceError(
                "retry scene_id does not match GPU receipt"
            )
        if shot_id != getattr(result, "shot_id", None):
            raise ProductionRetryEvidenceError(
                "retry shot_id does not match GPU receipt"
            )

        original_hash = _sha256(
            shot.get("original_request_hash"), field=f"{shot_id} original_request_hash"
        )
        accepted_hash = _sha256(
            shot.get("accepted_request_hash"), field=f"{shot_id} accepted_request_hash"
        )
        if accepted_hash != _sha256(
            getattr(result, "request_hash", None), field=f"{shot_id} GPU request_hash"
        ):
            raise ProductionRetryEvidenceError(
                "accepted retry request hash does not match GPU receipt"
            )

        attempts = shot.get("attempts")
        attempt_count = shot.get("attempt_count")
        if (
            not isinstance(attempt_count, int)
            or isinstance(attempt_count, bool)
            or attempt_count < 1
            or attempt_count > max_attempts
        ):
            raise ProductionRetryEvidenceError(f"{shot_id} attempt_count is invalid")
        if not isinstance(attempts, list) or len(attempts) != attempt_count:
            raise ProductionRetryEvidenceError(
                f"{shot_id} attempts do not match attempt_count"
            )

        transition_attempts = shot.get("transition_attempts", [])
        if not isinstance(transition_attempts, list):
            raise ProductionRetryEvidenceError(
                f"{shot_id} transition_attempts must be a list"
            )
        transition_by_index: dict[int, Mapping[str, Any]] = {}
        for transition in transition_attempts:
            if not isinstance(transition, Mapping):
                raise ProductionRetryEvidenceError(
                    f"{shot_id} transition attempt must be a mapping"
                )
            index = transition.get("attempt_index")
            if not isinstance(index, int) or isinstance(index, bool):
                raise ProductionRetryEvidenceError(
                    f"{shot_id} transition attempt index is invalid"
                )
            if index in transition_by_index:
                raise ProductionRetryEvidenceError(
                    f"{shot_id} contains duplicate transition attempt indices"
                )
            transition_by_index[index] = transition

        seen_request_hashes: set[str] = set()
        seen_output_hashes: set[str] = set()
        root_seed: int | None = None
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, Mapping):
                raise ProductionRetryEvidenceError(
                    f"{shot_id} attempt must be a mapping"
                )
            if attempt.get("attempt_index") != index:
                raise ProductionRetryEvidenceError(
                    f"{shot_id} retry attempt indices are not contiguous"
                )
            if (
                _sha256(
                    attempt.get("original_request_hash"),
                    field=f"{shot_id} attempt original_request_hash",
                )
                != original_hash
            ):
                raise ProductionRetryEvidenceError(
                    f"{shot_id} attempt changed original request lineage"
                )
            request_hash = _sha256(
                attempt.get("effective_request_hash"),
                field=f"{shot_id} attempt effective_request_hash",
            )
            output_hash = _sha256(
                attempt.get("output_sha256"), field=f"{shot_id} attempt output_sha256"
            )
            if request_hash in seen_request_hashes:
                raise ProductionRetryEvidenceError(
                    f"{shot_id} reused a request hash across retry attempts"
                )
            if output_hash in seen_output_hashes:
                raise ProductionRetryEvidenceError(
                    f"{shot_id} reused an artifact hash across retry attempts"
                )
            seen_request_hashes.add(request_hash)
            seen_output_hashes.add(output_hash)

            seed = attempt.get("seed")
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ProductionRetryEvidenceError(f"{shot_id} retry seed is invalid")
            if root_seed is None:
                root_seed = seed
            expected_seed = root_seed + seed_stride * index
            if seed != expected_seed:
                raise ProductionRetryEvidenceError(
                    f"{shot_id} retry seed progression does not match policy"
                )
            if index == 0 and request_hash != original_hash:
                raise ProductionRetryEvidenceError(
                    f"{shot_id} first attempt does not use original request hash"
                )

            is_final = index == attempt_count - 1
            transition = transition_by_index.get(index)
            shot_accepted = attempt.get("accepted") is True
            seam_accepted = transition is None or transition.get("accepted") is True
            if is_final:
                if not shot_accepted or not seam_accepted:
                    raise ProductionRetryEvidenceError(
                        f"{shot_id} final attempt was not fully accepted"
                    )
                if request_hash != accepted_hash:
                    raise ProductionRetryEvidenceError(
                        f"{shot_id} final retry request is not the accepted GPU request"
                    )
                if output_hash != _sha256(
                    getattr(receipt, "output_sha256", None),
                    field=f"{shot_id} accepted GPU output_sha256",
                ):
                    raise ProductionRetryEvidenceError(
                        f"{shot_id} final retry artifact is not the accepted GPU artifact"
                    )
            elif shot_accepted and seam_accepted:
                raise ProductionRetryEvidenceError(
                    f"{shot_id} contains an accepted attempt before the final attempt"
                )

        if shot_position == 0 and transition_attempts:
            raise ProductionRetryEvidenceError(
                "first connected shot must not carry predecessor transition attempts"
            )
        accepted_transition = shot.get("accepted_transition")
        if shot_position > 0 and gate.get("transition_gate_applied") is True:
            if not isinstance(accepted_transition, Mapping):
                raise ProductionRetryEvidenceError(
                    f"{shot_id} accepted transition evidence is missing"
                )
            if accepted_transition.get("accepted") is not True:
                raise ProductionRetryEvidenceError(
                    f"{shot_id} accepted transition did not pass"
                )
            if accepted_transition.get("attempt_index") != attempt_count - 1:
                raise ProductionRetryEvidenceError(
                    f"{shot_id} accepted transition is not bound to final retry attempt"
                )

        total_attempts += attempt_count
        rejected = attempt_count - 1
        total_rejections += rejected
        if rejected:
            recovered_shots.append(shot_id)

    expected_transitions = max(0, len(receipts) - 1)
    applied = gate.get("transition_gate_applied") is True
    accepted_transitions = gate.get("accepted_transitions")
    accepted_transition_count = gate.get("accepted_transition_count")
    if applied:
        if accepted_transition_count != expected_transitions:
            raise ProductionRetryEvidenceError(
                "accepted transition count does not cover every connected boundary"
            )
        if (
            not isinstance(accepted_transitions, list)
            or len(accepted_transitions) != expected_transitions
        ):
            raise ProductionRetryEvidenceError(
                "accepted transition list does not cover every connected boundary"
            )

    return {
        "schema": "cineos-production-retry-lineage-validation/0.1",
        "verified": True,
        "shot_count": len(receipts),
        "total_attempts": total_attempts,
        "rejected_attempts": total_rejections,
        "recovered_shot_ids": recovered_shots,
        "transition_gate_applied": applied,
    }


__all__ = [
    "ProductionRetryEvidenceError",
    "validate_production_quality_retry_gate",
]
