from dataclasses import dataclass, field


@dataclass(slots=True)
class GestureTrack:
    character_id: str
    gesture_type: str
    hand: str = "none"
    start_time: float = 0.0
    end_time: float = 0.0
    intensity: float = 0.5
    direction: str = ""
    target: str = ""
    repeat_policy: str = "once"
    prop_interaction: str = ""
    continuity_requirements: dict[str, object] = field(default_factory=dict)
    locked: bool = False
