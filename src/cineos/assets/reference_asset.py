"""Generic reference asset."""

from dataclasses import dataclass

from .base import Asset


@dataclass(slots=True, kw_only=True)
class GenericReference(Asset):
    asset_type = "reference"
