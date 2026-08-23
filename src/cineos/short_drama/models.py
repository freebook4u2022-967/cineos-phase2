"""Renderer-independent data contracts for short-drama planning."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class DramaBrief:
    premise: str
    duration_seconds: int = 180
    genre: str = "drama"
    tone: str = "cinematic"

    def __post_init__(self) -> None:
        if not self.premise.strip():
            raise ValueError("premise must not be empty")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")


@dataclass
class CharacterProfile:
    character_id: str
    name: str
    role: str
    motivation: str
    fear: str
    secret: str
    relationships: dict[str, str] = field(default_factory=dict)
    knowledge: list[str] = field(default_factory=list)
    emotion: str = "neutral"
    physical_state: str = "uninjured"
    wardrobe: str = "continuity-default"
    props: list[str] = field(default_factory=list)


@dataclass
class SceneState:
    scene_index: int
    location: str
    time_of_day: str
    weather: str
    characters: dict[str, dict] = field(default_factory=dict)
    environment: dict = field(default_factory=dict)


@dataclass
class DramaPlan:
    brief: DramaBrief
    story: dict = field(default_factory=dict)
    characters: list[CharacterProfile] = field(default_factory=list)
    screenplay: dict = field(default_factory=dict)
    direction: dict = field(default_factory=dict)
    shots: list[dict] = field(default_factory=list)
    continuity: dict = field(default_factory=dict)
    scene_states: list[SceneState] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a stable JSON-safe representation of the drama plan."""
        return asdict(self)
