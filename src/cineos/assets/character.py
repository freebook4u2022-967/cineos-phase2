"""Character assets."""

from dataclasses import dataclass

from .asset import Asset


@dataclass(slots=True, kw_only=True)
class Character(Asset):
    """A performer or fictional character."""
