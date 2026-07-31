"""Vehicle assets."""

from dataclasses import dataclass

from .asset import Asset


@dataclass(slots=True, kw_only=True)
class Vehicle(Asset):
    """A production vehicle."""
