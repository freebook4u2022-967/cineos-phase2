"""Motion identity descriptions."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class MotionProfile:
    walking_style: str = ""
    running_style: str = ""
    gesture_style: str = ""
    posture_behavior: str = ""
    combat_style: str = ""
    motion_reference_ids: list[str] = field(default_factory=list)
