from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EnvironmentConditioning:
    environment_asset_id: str
    approved_reference_ids: list[str]
    location_description: str = ""
    time_of_day: Any = None
    weather: Any = None
    lighting: Any = None
    atmosphere: Any = None
    spatial_continuity_constraints: dict[str, Any] = field(default_factory=dict)
