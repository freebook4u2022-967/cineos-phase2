"""Common audio cue vocabulary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4


class CueType(StrEnum):
    DIALOGUE = "dialogue"
    ROOM_TONE = "room_tone"
    AMBIENCE = "ambience"
    FOLEY = "foley"
    SOUND_EFFECT = "sound_effect"
    MUSIC = "music"
    TRANSITION = "transition"
    SILENCE = "silence"
    NARRATION = "narration"
    CROWD = "crowd"
    OFF_SCREEN_VOICE = "off_screen_voice"


@dataclass(slots=True)
class AudioCue:
    scene_id: str
    start_time: float
    duration: float
    cue_type: CueType | str
    cue_id: str = field(default_factory=lambda: str(uuid4()))
    shot_id: str | None = None
    asset_reference: str | None = None
    description: str = ""
    gain: float = 1.0
    pan: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0
    muted: bool = False
    solo: bool = False
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cue_type = CueType(self.cue_type)
        if self.start_time < 0 or self.duration < 0:
            raise ValueError("cue timing cannot be negative")
        if not -1 <= self.pan <= 1:
            raise ValueError("pan must be between -1 and 1")
