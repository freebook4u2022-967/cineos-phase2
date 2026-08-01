from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class WardrobeConditioning:
    wardrobe_asset_id: str
    garment_components: list[Any] = field(default_factory=list)
    continuity_locks: dict[str, Any] = field(default_factory=dict)
    allowed_variations: list[Any] = field(default_factory=list)
    scene_applicability: list[str] = field(default_factory=list)
    approved_reference_ids: list[str] = field(default_factory=list)
