"""Structured creative brief input."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CreativeBrief:
    title: str
    premise: str
    genre: str = "drama"
    tone: str = "cinematic"
    theme: str = ""
    language: str = "en"
    target_duration: float = 60.0
    target_audience: str = "general"
    visual_style: str = "naturalistic"
    narrative_constraints: list[str] = field(default_factory=list)
    required_characters: list[str] = field(default_factory=list)
    required_environments: list[str] = field(default_factory=list)
    forbidden_content: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.premise.strip():
            raise ValueError("title and premise are required")
        if self.target_duration <= 0:
            raise ValueError("target duration must be positive")
