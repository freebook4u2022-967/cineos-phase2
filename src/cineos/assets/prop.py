"""Prop assets."""

from dataclasses import dataclass

from .asset import Asset


@dataclass(slots=True, kw_only=True)
class Prop(Asset):
    """A physical story or set prop."""

    asset_type = "prop"
