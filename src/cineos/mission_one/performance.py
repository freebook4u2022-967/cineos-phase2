"""Explicit performance compilation without collapsing direction into prose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .brief import DirectedSceneBrief, DirectedShot


@dataclass(frozen=True, slots=True)
class PerformanceBrief:
    sections: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.sections)


def compile_performance(
    scene: DirectedSceneBrief, shot: DirectedShot
) -> PerformanceBrief:
    character = scene.character_ids[0] if scene.character_ids else "unknown subject"
    return PerformanceBrief(
        {
            "SUBJECT": {"character_id": character, "wardrobe": shot.wardrobe_state},
            "ACTION": {
                "primary": shot.action,
                "visible_behavior": shot.visible_behavior,
                "timing": f"within {shot.duration:g} seconds",
            },
            "PERFORMANCE": {
                "objective": shot.emotional_objective,
                "expression": shot.facial_expression,
                "posture": shot.body_posture,
                "gesture": shot.gesture,
                "eye_line": shot.eye_line,
                "blocking": shot.blocking,
            },
            "DIALOGUE": {
                "text": shot.dialogue_text,
                "delivery": shot.dialogue_delivery,
                "audio_strategy": "separate_audio",
                "lip_sync": "approximate_unless_measured",
            },
            "CAMERA": {
                "size": shot.shot_size,
                "angle": shot.camera_angle,
                "lens_intent": shot.lens_intent,
                "movement": shot.camera_movement,
            },
            "ENVIRONMENT": {
                "environment_id": scene.environment_id,
                "state": shot.environment_state,
                "props": shot.prop_state,
            },
            "LIGHTING": shot.lighting,
            "CONTINUITY": {
                "scene_locks": scene.continuity_locks,
                "previous_shot": shot.previous_shot_continuity,
            },
            "NEGATIVE CONSTRAINTS": [
                *scene.negative_constraints,
                *shot.forbidden_changes,
            ],
        }
    )
