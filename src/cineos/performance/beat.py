from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(slots=True)
class PerformanceBeat:
    character_id: str
    start_time: float
    end_time: float
    dramatic_purpose: str = ""
    emotional_objective: str = ""
    visible_action: str = ""
    reaction: str = ""
    subtext: str = ""
    intensity: float = 0.5
    restraint: float = 0.5
    pacing: str = "measured"
    priority: int = 0
    locked: bool = False
    manual: bool = False
    beat_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        if self.start_time < 0 or self.end_time < self.start_time:
            raise ValueError("invalid beat timing")
        if not 0 <= self.intensity <= 1 or not 0 <= self.restraint <= 1:
            raise ValueError("intensity and restraint must be between zero and one")

    @property
    def beat_uuid(self):
        return self.beat_id
