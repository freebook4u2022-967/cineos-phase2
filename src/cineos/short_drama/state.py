"""Continuity state engine for short-drama scenes."""

from __future__ import annotations

from copy import deepcopy

from .models import CharacterProfile, SceneState


class SceneStateEngine:
    """Track causal state across scenes without renderer-specific assumptions."""

    def initialize(self, characters: list[CharacterProfile]) -> SceneState:
        return SceneState(
            scene_index=0,
            location="unspecified",
            time_of_day="unspecified",
            weather="unspecified",
            characters={
                character.character_id: {
                    "location": "unspecified",
                    "emotion": character.emotion,
                    "physical_state": character.physical_state,
                    "wardrobe": character.wardrobe,
                    "props": list(character.props),
                    "knowledge": list(character.knowledge),
                }
                for character in characters
            },
            environment={},
        )

    def advance(self, previous: SceneState, scene: dict) -> SceneState:
        state = deepcopy(previous)
        state.scene_index = previous.scene_index + 1
        state.location = scene.get("location", previous.location)
        state.time_of_day = scene.get("time_of_day", previous.time_of_day)
        state.weather = scene.get("weather", previous.weather)
        for character_id, changes in scene.get("state_changes", {}).items():
            current = state.characters.setdefault(character_id, {})
            for key, value in changes.items():
                current[key] = deepcopy(value)
        state.environment.update(deepcopy(scene.get("environment_changes", {})))
        return state

    def build_timeline(
        self, characters: list[CharacterProfile], scenes: list[dict]
    ) -> list[SceneState]:
        timeline = [self.initialize(characters)]
        for scene in scenes:
            timeline.append(self.advance(timeline[-1], scene))
        return timeline
