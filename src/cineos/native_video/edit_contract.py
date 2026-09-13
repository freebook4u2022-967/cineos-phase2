"""Shared authored edit-contract helpers for native final-film validation.

The production pipeline exposes more than one final-film gate while callers migrate
between legacy and canonical acceptance paths. Scene-boundary interpretation must
therefore live in one implementation: persisted plans cannot be accepted by one gate
and rejected by another because transition aliases, serialized booleans, or timeline
validation drifted apart.

This module is deliberately renderer-neutral. It only interprets authored shot-plan
metadata and returns deterministic scene-boundary sample points for measured QC.
Invalid or ambiguous metadata fails closed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .boundary_eval import SceneBoundaryPoint

_SUPPORTED_TRANSITIONS = frozenset({"cut", "match", "fade"})
_TRANSITION_ALIASES = {
    "hard_cut": "cut",
    "hard-cut": "cut",
    "crossfade": "fade",
    "cross_fade": "fade",
    "match_cut": "match",
    "match-cut": "match",
}


def shot_payload(shot: Any) -> Mapping[str, Any]:
    """Return mapping payload metadata or an empty immutable-compatible mapping."""

    payload = getattr(shot, "payload", None)
    return payload if isinstance(payload, Mapping) else {}


def metadata_flag(payload: Mapping[str, Any], name: str) -> bool:
    """Decode persisted boolean metadata without truthiness ambiguity.

    Plans cross JSON/YAML/CLI/checkpoint boundaries, so strings such as ``"false"``
    must never become true merely because they are non-empty. Conventional serialized
    boolean forms remain supported for backwards compatibility; ambiguous values are
    rejected instead of silently changing authored edit semantics.
    """

    raw = payload.get(name, False)
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int) and raw in {0, 1}:
        return bool(raw)
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise ValueError(f"{name} must be boolean metadata; got {raw!r}")


def transition_for_boundary(outgoing: Any, incoming: Any) -> str:
    """Resolve an authored scene transition deterministically and fail closed.

    Incoming hard-cut/reset flags take precedence because they describe how the new
    scene enters. Transition metadata then resolves from incoming to outgoing hints,
    preserving legacy keys and aliases used by persisted plans.
    """

    incoming_payload = shot_payload(incoming)
    outgoing_payload = shot_payload(outgoing)

    if metadata_flag(incoming_payload, "continuity_reset") or metadata_flag(
        incoming_payload, "hard_cut"
    ):
        return "cut"

    raw = incoming_payload.get("transition_in")
    if raw is None:
        raw = incoming_payload.get("scene_transition")
    if raw is None:
        raw = incoming_payload.get("transition")
    if raw is None:
        raw = outgoing_payload.get("transition_out")
    if raw is None:
        raw = outgoing_payload.get("scene_transition")
    if raw is None:
        raw = outgoing_payload.get("transition")
    if raw is None:
        return "cut"

    transition = str(raw).strip().lower()
    transition = _TRANSITION_ALIASES.get(transition, transition)
    if transition not in _SUPPORTED_TRANSITIONS:
        # Preserve both long-standing public error phrases while legacy and
        # canonical final-film gates share this one implementation.
        raise ValueError(
            f"unsupported planned scene transition {raw!r}; unsupported scene "
            "transition; expected cut, match, or fade"
        )
    return transition


def validate_shot_identity_and_duration(shot: Any) -> tuple[str, float]:
    """Validate one planned shot and return normalized scene identity and duration."""

    scene_id = str(getattr(shot, "scene_id", "")).strip()
    if not scene_id:
        raise ValueError("planned shot is missing scene_id")

    duration = float(getattr(shot, "duration", 0.0))
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("planned shot duration must be finite and positive")
    return scene_id, duration


def planned_scene_boundaries(plan: Sequence[Any]) -> tuple[SceneBoundaryPoint, ...]:
    """Derive strictly ordered scene-boundary timestamps from an authored shot plan."""

    if not plan:
        raise ValueError("final-film quality gate requires a non-empty shot plan")

    elapsed = 0.0
    boundaries: list[SceneBoundaryPoint] = []
    previous_scene: str | None = None
    previous_shot: Any | None = None

    for shot in plan:
        scene_id, duration = validate_shot_identity_and_duration(shot)

        if previous_scene is not None and scene_id != previous_scene:
            boundaries.append(
                SceneBoundaryPoint(
                    from_scene_id=previous_scene,
                    to_scene_id=scene_id,
                    boundary_seconds=elapsed,
                    transition=transition_for_boundary(previous_shot, shot),
                )
            )

        next_elapsed = elapsed + duration
        if not math.isfinite(next_elapsed):
            raise ValueError("planned shot timeline must remain finite")
        elapsed = next_elapsed
        previous_scene = scene_id
        previous_shot = shot

    return tuple(boundaries)


def planned_duration_seconds(plan: Sequence[Any]) -> float:
    """Return validated authored runtime for duration/audio integrity gates."""

    if not plan:
        raise ValueError("final-film quality gate requires a non-empty shot plan")

    total = 0.0
    for shot in plan:
        _, duration = validate_shot_identity_and_duration(shot)
        total += duration
        if not math.isfinite(total):
            raise ValueError("planned shot timeline must remain finite")
    return total


__all__ = [
    "metadata_flag",
    "planned_duration_seconds",
    "planned_scene_boundaries",
    "shot_payload",
    "transition_for_boundary",
    "validate_shot_identity_and_duration",
]
