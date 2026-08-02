from dataclasses import dataclass, field

STANDARD_EXPRESSIONS = frozenset(
    {
        "neutral",
        "smile",
        "sadness",
        "anger",
        "fear",
        "surprise",
        "disgust",
        "pain",
        "concentration",
    }
)


@dataclass(slots=True)
class FacialKeyframe:
    time: float
    expression: str = "neutral"
    intensity: float = 0.0
    brow_movement: float = 0.0
    cheek_tension: float = 0.0
    jaw_tension: float = 0.0
    gaze_direction: str = "forward"
    transition_duration: float = 0.0
    blink: bool = False


@dataclass(slots=True)
class FacialPerformanceTrack:
    character_id: str
    keyframes: list[FacialKeyframe] = field(default_factory=list)
    blink_times: list[float] = field(default_factory=list)
    approved_custom_expressions: list[str] = field(default_factory=list)
    locked: bool = False

    def validate_expressions(self):
        approved = STANDARD_EXPRESSIONS | set(self.approved_custom_expressions)
        return [k.expression for k in self.keyframes if k.expression not in approved]
