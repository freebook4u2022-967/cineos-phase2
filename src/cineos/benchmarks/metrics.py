"""Typed benchmark measurements."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class MetricStatus(StrEnum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    MANUALLY_REVIEWED = "manually_reviewed"


METRIC_NAMES = (
    "execution_success",
    "render_completion_rate",
    "validation_pass_rate",
    "identity_score",
    "wardrobe_continuity_score",
    "prop_continuity_score",
    "environment_continuity_score",
    "temporal_stability",
    # Direct difficult-case measurements. These intentionally complement broad
    # temporal/continuity scores so competitive cases cannot pass on proxies alone.
    "anatomy_integrity_score",
    "contact_consistency_score",
    "locomotion_coherence_score",
    "camera_motion_coherence_score",
    "physics_consistency_score",
    "lip_sync_timing_accuracy",
    "audio_alignment",
    "final_assembly_success",
    "output_duration_accuracy",
    "output_resolution",
    "output_fps",
    "peak_vram",
    "peak_ram",
    "runtime_per_shot",
    "total_build_time",
    "disk_usage",
    "recovery_attempts",
    "manual_review_count",
)


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    value: Any = None
    status: MetricStatus = MetricStatus.UNAVAILABLE
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.name not in METRIC_NAMES:
            raise ValueError(f"unknown benchmark metric: {self.name}")
        if (
            self.status in {MetricStatus.MEASURED, MetricStatus.ESTIMATED}
            and self.value is None
        ):
            raise ValueError("measured and estimated metrics require a value")
