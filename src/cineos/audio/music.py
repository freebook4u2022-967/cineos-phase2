"""Music planning models; generation belongs to provider adapters."""

from dataclasses import dataclass, field
from enum import StrEnum
from uuid import uuid4


class MusicSourceType(StrEnum):
    USER_PROVIDED = "user-provided"
    LICENSED_ASSET = "licensed-asset"
    ORIGINAL_GENERATED = "original-generated"
    LIBRARY_ASSET = "library-asset"
    NONE = "none"


@dataclass(slots=True)
class MusicCue:
    scene_id: str
    start_time: float
    end_time: float
    dramatic_purpose: str
    mood: str
    source_type: MusicSourceType | str = MusicSourceType.NONE
    cue_id: str = field(default_factory=lambda: str(uuid4()))
    tempo: float | None = None
    key: str | None = None
    intensity_curve: list[dict[str, float]] = field(default_factory=list)
    ducking_rules: dict[str, object] = field(default_factory=dict)
    fade_rules: dict[str, float] = field(default_factory=dict)
    rights_metadata: dict[str, object] = field(default_factory=dict)
    approved_asset_reference: str | None = None

    def __post_init__(self) -> None:
        self.source_type = MusicSourceType(self.source_type)
        if self.start_time < 0 or self.end_time < self.start_time:
            raise ValueError("invalid music cue timing")
        if self.source_type != MusicSourceType.NONE and not self.rights_metadata:
            raise ValueError("music cues require rights metadata")

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
