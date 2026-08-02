"""Continuity state and deterministic propagation."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ContinuityState:
    character_identity: dict[str, str] = field(default_factory=dict)
    wardrobe: dict[str, str] = field(default_factory=dict)
    props: dict[str, str] = field(default_factory=dict)
    environment: str = ""
    lighting: str = ""
    weather: str = ""
    time_of_day: str = ""
    injuries: dict[str, str] = field(default_factory=dict)
    emotional_state: dict[str, str] = field(default_factory=dict)
    spatial_position: dict[str, str] = field(default_factory=dict)
    dialogue_state: str = ""
    previous_shot_carry_over: dict[str, Any] = field(default_factory=dict)

    def carry(self, **changes: Any) -> "ContinuityState":
        from dataclasses import asdict

        values = asdict(self)
        values.update(changes)
        return ContinuityState(**values)


ContinuityPlan = ContinuityState
