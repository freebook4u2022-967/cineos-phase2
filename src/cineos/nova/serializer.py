"""Canonical NOVA plan persistence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cineos.core import (
    Character,
    Environment,
    MovieProject,
    Prop,
    Scene,
    Shot,
    Timeline,
)

from .brief import CreativeBrief
from .continuity import ContinuityState
from .director import DirectorPlan
from .pacing import PacingPlan
from .performance_plan import PerformancePlan
from .scene_plan import ScenePlan
from .shot_plan import ShotPlan
from .story import StoryPlan


def plan_to_dict(plan: DirectorPlan) -> dict[str, Any]:
    """Return compiler-loadable project JSON enriched with a NOVA section."""
    project = plan.project
    shot_plans = []
    for item in plan.shots:
        shot = asdict(item)
        shot["renderer_capability_requirements"] = sorted(
            item.renderer_capability_requirements
        )
        shot_plans.append(shot)
    return {
        "format": "cineos-nova-plan-1",
        "title": project.title,
        "author": project.author,
        "version": project.version,
        "fps": project.fps,
        "resolution": list(project.resolution),
        "aspect_ratio": project.aspect_ratio,
        "characters": [asdict(item) for item in project.characters],
        "locations": [asdict(item) for item in project.locations],
        "props": [asdict(item) for item in project.props],
        "scenes": [asdict(item) for item in project.scenes],
        "timeline": asdict(project.timeline),
        "nova": {
            "brief": asdict(plan.brief),
            "story": {**asdict(plan.story), "content_hash": plan.story.content_hash},
            "scene_plans": [asdict(item) for item in plan.scenes],
            "shot_plans": shot_plans,
            "planner_id": plan.planner_id,
            "planner_version": plan.planner_version,
            "seed": plan.seed,
            "revision_history": plan.revision_history,
        },
    }


def serialize(plan: DirectorPlan) -> str:
    return json.dumps(plan_to_dict(plan), sort_keys=True, separators=(",", ":"))


def save(plan: DirectorPlan, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize(plan) + "\n", encoding="utf-8")
    return path


def _continuity(value: dict[str, Any]) -> ContinuityState:
    return ContinuityState(**value)


def plan_from_dict(value: dict[str, Any]) -> DirectorPlan:
    nova = value["nova"]
    story_value = dict(nova["story"])
    story_value.pop("content_hash", None)
    scenes = []
    for raw in nova["scene_plans"]:
        raw = dict(raw)
        raw["continuity_inputs"] = _continuity(raw["continuity_inputs"])
        raw["continuity_outputs"] = _continuity(raw["continuity_outputs"])
        raw["pacing"] = PacingPlan(**raw["pacing"])
        scenes.append(ScenePlan(**raw))
    shots = []
    for raw in nova["shot_plans"]:
        raw = dict(raw)
        raw["performance_direction"] = PerformancePlan(**raw["performance_direction"])
        raw["renderer_capability_requirements"] = set(
            raw["renderer_capability_requirements"]
        )
        shots.append(ShotPlan(**raw))
    timeline_value = value.get("timeline", {})
    project = MovieProject(
        value["title"],
        value.get("author", ""),
        value.get("version", "1.0"),
        float(value.get("fps", 24)),
        tuple(value.get("resolution", [1920, 1080])),
        value.get("aspect_ratio", "16:9"),
        [Character(**item) for item in value.get("characters", [])],
        [Environment(**item) for item in value.get("locations", [])],
        [Prop(**item) for item in value.get("props", [])],
        [
            Scene(**{**item, "shots": [Shot(**shot) for shot in item.get("shots", [])]})
            for item in value.get("scenes", [])
        ],
        Timeline(**timeline_value),
    )
    return DirectorPlan(
        CreativeBrief(**nova["brief"]),
        StoryPlan(**story_value),
        scenes,
        shots,
        project,
        nova["planner_id"],
        nova["planner_version"],
        int(nova["seed"]),
        list(nova.get("revision_history", [])),
    )


def load(source: str | Path | dict[str, Any]) -> DirectorPlan:
    if isinstance(source, dict):
        value = source
    else:
        source_path = Path(source)
        value = json.loads(source_path.read_text(encoding="utf-8"))
    return plan_from_dict(value)
