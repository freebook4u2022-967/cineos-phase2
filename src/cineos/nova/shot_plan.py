"""Shot-level directing plan."""

from dataclasses import dataclass, field

from .camera_plan import CameraPlan
from .performance_plan import PerformancePlan


@dataclass(slots=True)
class ShotPlan:
    shot_id: str
    scene_id: str
    shot_purpose: str
    action: str
    dialogue_intent: str = ""
    character_blocking: dict[str, str] = field(default_factory=dict)
    framing: str = "medium shot"
    angle: str = "eye level"
    lens: str = "50mm"
    camera_movement: str = "static"
    focus_target: str = "principal action"
    lighting_intent: str = "motivated natural light"
    performance_direction: PerformancePlan = field(default_factory=PerformancePlan)
    duration: float = 4.0
    transition: str = "cut"
    continuity_constraints: dict[str, object] = field(default_factory=dict)
    renderer_capability_requirements: set[str] = field(default_factory=set)
    rationale: str = ""

    @property
    def camera(self) -> CameraPlan:
        return CameraPlan(
            self.framing, self.angle, self.lens, self.camera_movement, self.focus_target
        )
