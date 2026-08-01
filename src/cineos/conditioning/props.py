from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PropConditioning:
    asset_id: str
    ownership_or_character_relationship: str = ""
    spatial_role: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    continuity_locks: dict[str, Any] = field(default_factory=dict)
    approved_reference_ids: list[str] = field(default_factory=list)


VehicleConditioning = PropConditioning
