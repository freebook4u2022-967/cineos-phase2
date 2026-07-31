"""Approved wardrobe identity profiles."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class WardrobeProfile:
    wardrobe_asset_id: str
    scene_applicability: list[str] = field(default_factory=list)
    garment_components: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    accessories: list[str] = field(default_factory=list)
    continuity_lock: bool = False
    allowed_variations: dict[str, Any] = field(default_factory=dict)
