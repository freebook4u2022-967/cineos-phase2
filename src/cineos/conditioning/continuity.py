from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ContinuityConditioning:
    previous_shot_id: str | None = None
    next_shot_id: str | None = None
    persistent_character_state: dict[str, Any] = field(default_factory=dict)
    wardrobe_state: dict[str, Any] = field(default_factory=dict)
    prop_state: dict[str, Any] = field(default_factory=dict)
    environment_state: dict[str, Any] = field(default_factory=dict)
    lighting_continuity: dict[str, Any] = field(default_factory=dict)
    forbidden_changes: list[str] = field(default_factory=list)
    required_carry_over_references: list[str] = field(default_factory=list)
