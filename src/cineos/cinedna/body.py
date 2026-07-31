"""Renderer-independent body descriptors."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class BodyProfile:
    height: str = ""
    body_proportions: str = ""
    build: str = ""
    posture: str = ""
    dominant_hand: str = ""
    movement_notes: list[str] = field(default_factory=list)
    silhouette_constraints: list[str] = field(default_factory=list)
