from dataclasses import dataclass


@dataclass(slots=True)
class EyeLineTrack:
    character_id: str
    target_character_id: str | None = None
    target_object_id: str | None = None
    screen_direction: str = "center"
    start_time: float = 0.0
    end_time: float = 0.0
    gaze_shift_timing: float = 0.0
    focus_duration: float = 0.0
    reaction_delay: float = 0.0
    continuity_carry_over: bool = False
    locked: bool = False
