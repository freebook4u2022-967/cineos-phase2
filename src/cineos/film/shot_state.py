"""Persistent state for one renderable shot."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ShotState:
    shot_id: str
    render_job_id: str | None = None
    conditioning_package_id: str | None = None
    attempt_count: int = 0
    output_path: str | None = None
    render_status: str = "pending"
    validation_status: str = "pending"
    recovery_status: str = "not_required"
    selected_output: str | None = None
    warnings: list[str] = field(default_factory=list)
    failure_reason: str | None = None
    timing_metrics: dict[str, float] = field(default_factory=dict)
    attempt_history: list[dict[str, Any]] = field(default_factory=list)
    output_hash: str | None = None

    @property
    def approved(self) -> bool:
        return self.validation_status == "approved" and bool(self.selected_output)
