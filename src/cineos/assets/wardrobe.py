"""Wardrobe assets."""

from dataclasses import dataclass

from .asset import Asset


@dataclass(slots=True, kw_only=True)
class Wardrobe(Asset):
    """A costume or wardrobe item."""
