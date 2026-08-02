"""Deterministic, provider-neutral NOVA directing engine."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from cineos.assets import AssetRegistry
from cineos.atlas import RendererCapabilities
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
from .camera_plan import Framing, Movement
from .continuity import ContinuityState
from .exceptions import MissingAssetError, PlannerNotFoundError
from .pacing import PacingPlan
from .performance_plan import PerformancePlan
from .scene_plan import ScenePlan
from .shot_plan import ShotPlan
from .story import StoryPlan
from .validator import NOVAValidator


@dataclass(slots=True)
class DirectorPlan:
    brief: CreativeBrief
    story: StoryPlan
    scenes: list[ScenePlan]
    shots: list[ShotPlan]
    project: MovieProject
    planner_id: str
    planner_version: str
    seed: int
    revision_history: list[dict[str, Any]] = field(default_factory=list)


class PlanningProvider(ABC):
    """Interface implemented by local rules or optional language-model providers."""

    planner_id = "abstract"
    version = "0"

    @abstractmethod
    def create(
        self,
        brief: CreativeBrief,
        assets: dict[str, Any],
        *,
        seed: int,
        max_scenes: int | None,
        max_shots: int | None,
    ) -> tuple[StoryPlan, list[ScenePlan], list[ShotPlan]]:
        """Return renderer-neutral story, scene, and shot plans."""


class RuleBasedPlanner(PlanningProvider):
    """Offline planner whose output is stable for identical inputs."""

    planner_id = "rule-based"
    version = "1.0"

    def create(self, brief, assets, *, seed, max_scenes, max_shots):
        rng = random.Random(seed)
        characters = list(brief.required_characters)
        environments = list(brief.required_environments)
        scene_count = max(1, min(max_scenes or 3, max_shots or 3, 3))
        shot_limit = max_shots or scene_count * 3
        beats = ["setup", "confrontation", "resolution"][:scene_count]
        acts = [
            {"act": str(index + 1), "objective": beat}
            for index, beat in enumerate(beats)
        ]
        lead = characters[0] if characters else "the protagonist"
        story = StoryPlan(
            logline=f"{lead} must {brief.premise.rstrip('.').lower()}.",
            synopsis=f"A {brief.tone} {brief.genre} about {brief.premise}",
            act_structure=acts,
            dramatic_objective=brief.premise,
            central_conflict=f"Obstacles challenge {lead}'s objective.",
            turning_points=[f"The {beat} changes the objective." for beat in beats[1:]],
            climax=f"{lead} makes a decisive choice.",
            resolution=f"The consequences reveal the theme: {brief.theme or 'change'}.",
            character_arcs={
                item: "moves from uncertainty to purposeful action"
                for item in characters
            },
            estimated_duration=brief.target_duration,
            locked_constraints=list(brief.narrative_constraints),
            rationale="A three-beat structure gives the production clear escalation.",
        )
        scene_duration = brief.target_duration / scene_count
        scenes: list[ScenePlan] = []
        shots: list[ShotPlan] = []
        state = ContinuityState(
            character_identity={item: item for item in characters},
            wardrobe={item: "established wardrobe" for item in characters},
            environment=environments[0] if environments else "",
            time_of_day=str(brief.metadata.get("time_of_day", "day")),
            emotional_state={item: "focused" for item in characters},
        )
        patterns = [
            (Framing.ESTABLISHING, Movement.STATIC),
            (Framing.MEDIUM, Movement.TRACKING),
            (Framing.CLOSE_UP, Movement.PUSH_IN),
        ]
        remaining = shot_limit
        for index, beat in enumerate(beats):
            scene_id = f"scene-{index + 1:03d}"
            location = environments[index % len(environments)] if environments else ""
            output = state.carry(
                environment=location,
                emotional_state={item: beat for item in characters},
            )
            scenes_left = scene_count - index
            count = min(3, max(1, remaining // scenes_left))
            remaining -= count
            scene = ScenePlan(
                scene_id=scene_id,
                title=beat.title(),
                narrative_purpose=f"Deliver the {beat} beat",
                dramatic_beat=beat,
                location_asset_id=location,
                participating_character_ids=characters,
                emotional_state=output.emotional_state,
                start_condition="Carries the previous scene state",
                end_condition=f"The {beat} changes the dramatic question",
                estimated_duration=scene_duration,
                continuity_inputs=state,
                continuity_outputs=output,
                pacing=PacingPlan(
                    scene_rhythm=("measured", "building", "decisive")[index],
                    shot_duration_target=scene_duration / max(count, 1),
                    escalation=(index + 1) / scene_count,
                    emotional_intensity=(index + 1) / scene_count,
                ),
                rationale=(
                    f"Scene {index + 1} advances the {beat} without adding assets."
                ),
            )
            scenes.append(scene)
            for shot_index in range(count):
                framing, movement = patterns[(shot_index + index) % len(patterns)]
                duration = scene_duration / count
                shots.append(
                    ShotPlan(
                        shot_id=f"{scene_id}-shot-{shot_index + 1:03d}",
                        scene_id=scene_id,
                        shot_purpose=("orient", "develop", "reveal")[shot_index],
                        action=f"{lead} advances the {beat} objective",
                        character_blocking={
                            item: "maintains established screen position"
                            for item in characters
                        },
                        framing=framing,
                        camera_movement=movement,
                        focus_target=lead,
                        performance_direction=PerformancePlan(
                            emotional_objective=f"achieve the {beat} objective",
                            tempo=("measured", "urgent", "decisive")[index],
                            subtext=brief.theme or "change",
                        ),
                        duration=duration,
                        continuity_constraints={
                            "input_scene": scene_id,
                            "environment": location,
                        },
                        rationale=(
                            f"The {framing} communicates the {beat} beat economically."
                        ),
                    )
                )
            state = output
            if remaining <= 0:
                break
        # Make version and seed explicit deterministic inputs.
        rng.random()
        return story, scenes, shots


class NOVADirector:
    def __init__(self, asset_registry: AssetRegistry | None = None) -> None:
        self.asset_registry = asset_registry or AssetRegistry()
        self.providers: dict[str, PlanningProvider] = {}
        self.register_provider(RuleBasedPlanner())
        self.last_plan: DirectorPlan | None = None

    def register_provider(self, provider: PlanningProvider) -> None:
        self.providers[provider.planner_id] = provider

    def _resolve_assets(self, brief: CreativeBrief) -> dict[str, Any]:
        available = {str(item.asset_id): item for item in self.asset_registry.list()}
        by_name = {item.name: item for item in self.asset_registry.list()}
        resolved: dict[str, Any] = {}
        for reference in brief.required_characters + brief.required_environments:
            asset = available.get(reference) or by_name.get(reference)
            if asset is None:
                raise MissingAssetError(
                    f"required approved asset is unavailable: {reference}"
                )
            resolved[reference] = asset
        return resolved

    def create_plan(
        self,
        brief: CreativeBrief,
        *,
        seed: int = 0,
        planner: str = "rule-based",
        max_scenes: int | None = None,
        max_shots: int | None = None,
        renderer_capabilities: RendererCapabilities | None = None,
    ) -> DirectorPlan:
        try:
            provider = self.providers[planner]
        except KeyError as error:
            raise PlannerNotFoundError(planner) from error
        assets = self._resolve_assets(brief)
        story, scenes, shots = provider.create(
            brief, assets, seed=seed, max_scenes=max_scenes, max_shots=max_shots
        )
        project = self._movie_project(brief, scenes, shots, assets)
        result = DirectorPlan(
            brief, story, scenes, shots, project, planner, provider.version, seed
        )
        NOVAValidator(renderer_capabilities).validate(result)
        self.last_plan = result
        return result

    def direct(self, brief: CreativeBrief, **options: Any) -> MovieProject:
        """Produce the compiler-compatible core MovieProject."""
        return self.create_plan(brief, **options).project

    plan = create_plan

    @staticmethod
    def _movie_project(brief, scenes, shots, assets):
        core_scenes = []
        timeline = Timeline()
        for scene_plan in scenes:
            scene_shots = [
                item for item in shots if item.scene_id == scene_plan.scene_id
            ]
            core = Scene(
                scene_plan.scene_id,
                scene_plan.title,
                scene_plan.narrative_purpose,
                [
                    Shot(
                        item.shot_id,
                        item.framing,
                        item.lens,
                        item.camera_movement,
                        item.lighting_intent,
                        item.action,
                        item.dialogue_intent,
                        item.duration,
                    )
                    for item in scene_shots
                ],
                scene_plan.location_asset_id or None,
                list(scene_plan.participating_character_ids),
                sum(item.duration for item in scene_shots),
            )
            core_scenes.append(core)
            timeline.add_scene(core.scene_id)
            for shot in core.shots:
                timeline.add_shot(core.scene_id, shot.shot_id)
        production = list(assets.values())
        characters = [
            Character(str(a.asset_id), a.name, a.description)
            for a in production
            if a.kind == "character"
        ]
        locations = [
            Environment(str(a.asset_id), a.name, a.description)
            for a in production
            if a.kind == "environment"
        ]
        props = [
            Prop(str(a.asset_id), a.name, a.description)
            for a in production
            if a.kind == "prop"
        ]
        registry = AssetRegistry()
        for asset in production:
            registry.register(asset)
        return MovieProject(
            title=brief.title,
            author=str(brief.metadata.get("author", "NOVA")),
            characters=characters,
            locations=locations,
            props=props,
            scenes=core_scenes,
            timeline=timeline,
            asset_registry=registry,
            asset_ids=[a.asset_id for a in production],
        )
