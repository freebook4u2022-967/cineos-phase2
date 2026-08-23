"""Renderer-independent data contracts for short-drama planning."""

from dataclasses import dataclass, field


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
class DramaPlan:
    brief: DramaBrief
    story: dict = field(default_factory=dict)
    screenplay: dict = field(default_factory=dict)
    direction: dict = field(default_factory=dict)
    shots: list[dict] = field(default_factory=list)
    continuity: dict = field(default_factory=dict)
