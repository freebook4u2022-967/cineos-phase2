from dataclasses import asdict, dataclass, field
from typing import Any

STATUSES = {"measured", "visually reviewed", "unsupported", "unavailable", "failed"}


@dataclass
class MissionOneReport:
    render_success: dict[str, str] = field(default_factory=dict)
    action_compliance: str = "unavailable"
    dialogue_performance: str = "unavailable"
    camera_compliance: str = "unavailable"
    continuity_warnings: list[str] = field(default_factory=list)
    identity_limitations: str = "unsupported"
    lip_sync_limitations: str = "unsupported"
    temporal_stability_notes: list[str] = field(default_factory=list)
    final_assembly_status: str = "unavailable"
    render_time: float | None = None
    model: str = ""
    hardware: str = ""
    manual_review_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
