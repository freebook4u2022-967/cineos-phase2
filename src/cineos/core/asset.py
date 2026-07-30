"""Asset value objects used by a movie project."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Asset:
    """A named production asset with a stable project-local identifier."""

    asset_id: str
    name: str
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Character(Asset):
    """A character appearing in scenes."""


@dataclass(slots=True)
class Environment(Asset):
    """A location/environment in which scenes take place."""


@dataclass(slots=True)
class Prop(Asset):
    """A physical story or set prop."""
