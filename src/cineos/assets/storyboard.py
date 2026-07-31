"""Storyboard assets."""

from dataclasses import dataclass

from .asset import Asset


@dataclass(slots=True, kw_only=True)
class Storyboard(Asset):
    """A storyboard sequence or panel collection."""
