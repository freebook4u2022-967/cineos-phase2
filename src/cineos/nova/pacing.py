"""Pacing intent for a scene or sequence."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class PacingPlan:
    scene_rhythm: str = "measured"
    shot_duration_target: float = 4.0
    escalation: float = 0.5
    pause_placement: list[str] = field(default_factory=list)
    reveal_timing: str = "turning point"
    action_density: float = 0.5
    emotional_intensity: float = 0.5
    transition_rhythm: str = "continuous"


PacingModel = PacingPlan
