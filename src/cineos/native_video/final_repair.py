"""Deterministic remediation planning for rejected assembled native films.

Final-film QC is intentionally evidence-driven and may reject for very different
reasons: a global temporal defect, an edit-boundary continuity break, container
length mismatch, or encoded-audio failure. Treating all of those as one opaque
failure leads to wasteful whole-film rerenders and makes autonomous recovery hard
to audit.

This module converts measured final-film evidence into a renderer-neutral repair
plan. It does not generate replacement pixels itself. Instead it identifies the
smallest safe remediation domain and, where evidence is scene-local, the concrete
shot IDs that should be regenerated on the next production recovery pass.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FinalFilmRepairAction:
    """One auditable remediation step derived from measured rejection evidence."""

    domain: str
    reason: str
    shot_ids: tuple[str, ...] = ()
    scene_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FinalFilmRepairPlan:
    """Aggregate recovery contract for a rejected final movie."""

    required: bool
    actions: tuple[FinalFilmRepairAction, ...]
    affected_shot_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scene_edges(plan: list[Any] | tuple[Any, ...]) -> dict[str, tuple[str, str]]:
    shots_by_scene: dict[str, list[str]] = {}
    for item in plan:
        scene_id = str(getattr(item, "scene_id", "")).strip()
        shot_id = str(getattr(item, "shot_id", "")).strip()
        if scene_id and shot_id:
            shots_by_scene.setdefault(scene_id, []).append(shot_id)
    return {
        scene_id: (shot_ids[0], shot_ids[-1])
        for scene_id, shot_ids in shots_by_scene.items()
        if shot_ids
    }


def _decision(component: Any) -> str:
    return str(getattr(component, "decision", "")).strip().lower()


def _directives(component: Any) -> tuple[str, ...]:
    return tuple(
        str(item) for item in (getattr(component, "directives", ()) or ()) if str(item)
    )


def build_final_film_repair_plan(
    *,
    plan: list[Any] | tuple[Any, ...],
    temporal: Any,
    boundaries: Any | None = None,
    duration: Any | None = None,
    audio: Any | None = None,
) -> FinalFilmRepairPlan:
    """Translate measured rejection evidence into minimal safe recovery actions.

    Scene-boundary defects are localized to the outgoing shot of the source scene
    and incoming shot of the destination scene. Global temporal defects are not
    safely localizable from aggregate sampling, so they deliberately request a
    timeline-level visual regeneration rather than guessing at a shot. Duration
    and audio failures stay in assembly/audio domains so healthy visual shots are
    not regenerated unnecessarily.
    """

    timeline = tuple(plan)
    scene_edges = _scene_edges(timeline)
    actions: list[FinalFilmRepairAction] = []

    if _decision(temporal) == "reject":
        reasons = _directives(temporal) or (
            "final temporal evaluation rejected the movie",
        )
        for reason in reasons:
            actions.append(
                FinalFilmRepairAction(domain="visual_timeline", reason=reason)
            )

    if boundaries is not None:
        for item in tuple(getattr(boundaries, "boundaries", ()) or ()):
            if _decision(item) != "reject":
                continue
            from_scene = str(getattr(item, "from_scene_id", "")).strip()
            to_scene = str(getattr(item, "to_scene_id", "")).strip()
            shot_ids: list[str] = []
            if from_scene in scene_edges:
                shot_ids.append(scene_edges[from_scene][1])
            if to_scene in scene_edges:
                incoming = scene_edges[to_scene][0]
                if incoming not in shot_ids:
                    shot_ids.append(incoming)
            reasons = _directives(item) or (
                "scene-boundary continuity evaluation rejected the edit",
            )
            for reason in reasons:
                actions.append(
                    FinalFilmRepairAction(
                        domain="scene_continuity",
                        reason=reason,
                        shot_ids=tuple(shot_ids),
                        scene_ids=tuple(
                            value for value in (from_scene, to_scene) if value
                        ),
                    )
                )

    if duration is not None and _decision(duration) == "reject":
        reasons = _directives(duration) or (
            "encoded duration does not match the authored timeline",
        )
        for reason in reasons:
            actions.append(FinalFilmRepairAction(domain="assembly", reason=reason))

    if audio is not None and _decision(audio) == "reject":
        reasons = _directives(audio) or (
            "encoded audio integrity evaluation rejected the movie",
        )
        for reason in reasons:
            actions.append(FinalFilmRepairAction(domain="audio", reason=reason))

    unique: list[FinalFilmRepairAction] = []
    seen: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
    for action in actions:
        key = (action.domain, action.reason, action.shot_ids, action.scene_ids)
        if key not in seen:
            seen.add(key)
            unique.append(action)

    affected: list[str] = []
    for action in unique:
        for shot_id in action.shot_ids:
            if shot_id not in affected:
                affected.append(shot_id)

    return FinalFilmRepairPlan(
        required=bool(unique),
        actions=tuple(unique),
        affected_shot_ids=tuple(affected),
    )


__all__ = [
    "FinalFilmRepairAction",
    "FinalFilmRepairPlan",
    "build_final_film_repair_plan",
]
