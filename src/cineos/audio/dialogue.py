"""Dialogue cue model."""

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(slots=True)
class DialogueCue:
    scene_id: str
    shot_id: str
    character_id: str
    line_text: str
    language: str
    start_time: float
    target_duration: float
    cue_id: str = field(default_factory=lambda: str(uuid4()))
    delivery_intent: str = ""
    emotional_state: str = ""
    pause_before: float = 0.0
    pause_after: float = 0.0
    overlap_rules: str = "disallow"
    subtitle_text: str = ""
    continuity_notes: str = ""
    approved_voice_profile_id: str | None = None
    gain: float = 1.0
    pan: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    muted: bool = False
    solo: bool = False

    def __post_init__(self) -> None:
        if (
            min(
                self.start_time,
                self.target_duration,
                self.pause_before,
                self.pause_after,
            )
            < 0
        ):
            raise ValueError("dialogue timing cannot be negative")

    @property
    def duration(self) -> float:
        return self.target_duration

    @property
    def end_time(self) -> float:
        return self.start_time + self.target_duration
