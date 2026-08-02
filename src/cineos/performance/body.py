from dataclasses import dataclass, field


@dataclass(slots=True)
class BodyPerformanceTrack:
    character_id: str
    posture: str = "neutral"
    weight_distribution: str = "balanced"
    torso_orientation: str = "forward"
    shoulder_state: str = "neutral"
    head_pose: str = "neutral"
    walking_style: str = "none"
    movement_speed: float = 0.0
    physical_tension: float = 0.0
    breathing_rhythm: str = "natural"
    spatial_blocking: list[dict[str, object]] = field(default_factory=list)
    entry_conditions: dict[str, object] = field(default_factory=dict)
    exit_conditions: dict[str, object] = field(default_factory=dict)
    locked: bool = False
