"""Auditable actor direction."""

from dataclasses import dataclass


@dataclass(slots=True)
class PerformancePlan:
    emotional_objective: str = "pursue the scene objective"
    visible_behavior: str = "controlled attention"
    body_language: str = "grounded"
    eye_line: str = "scene partner"
    gesture: str = "minimal"
    tempo: str = "measured"
    restraint_level: float = 0.5
    dialogue_delivery: str = "natural"
    reaction_beat: str = "register the change"
    subtext: str = "unspoken need"

    def __post_init__(self) -> None:
        if not 0 <= self.restraint_level <= 1:
            raise ValueError("restraint level must be between zero and one")


PerformanceDirection = PerformancePlan
