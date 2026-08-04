"""Canonical schemas for the first directed three-shot production."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DirectedShot:
    shot_id: str
    scene_id: str
    duration: float
    narrative_purpose: str
    action: str
    visible_behavior: str = ""
    dialogue_text: str = ""
    dialogue_delivery: str = ""
    emotional_objective: str = ""
    facial_expression: str = ""
    body_posture: str = ""
    gesture: str = ""
    eye_line: str = ""
    blocking: str = ""
    shot_size: str = "medium"
    camera_angle: str = "eye level"
    lens_intent: str = "natural perspective"
    camera_movement: str = "locked"
    lighting: str = ""
    environment_state: str = ""
    wardrobe_state: str = ""
    prop_state: str = ""
    previous_shot_continuity: dict[str, Any] = field(default_factory=dict)
    forbidden_changes: list[str] = field(default_factory=list)
    renderer_settings: dict[str, Any] = field(default_factory=dict)
    deterministic_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DirectedShot:
        return cls(**value)


@dataclass(slots=True)
class DirectedSceneBrief:
    title: str
    genre: str
    tone: str
    visual_style: str
    character_ids: list[str]
    environment_id: str
    story_objective: str
    scene_start_state: str
    scene_end_state: str
    dialogue: list[dict[str, Any]]
    target_duration: float
    continuity_locks: dict[str, Any] = field(default_factory=dict)
    negative_constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    scene_id: str = "scene-1"
    shots: list[DirectedShot] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DirectedSceneBrief:
        data = dict(value)
        data["shots"] = [DirectedShot.from_dict(item) for item in data.get("shots", [])]
        return cls(**data)
