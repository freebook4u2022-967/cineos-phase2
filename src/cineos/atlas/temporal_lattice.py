"""Deterministic temporal-frame planning for pretrained video foundations.

CINEOS owns shot timing even when generation is delegated to an external
pretrained foundation. Some video VAEs only accept frame counts on a temporal
lattice (for Wan, ``(frames - 1)`` must be divisible by the temporal compression
ratio). Planning that adjustment before inference avoids hidden duration drift
inside the borrowed pipeline and makes trimming explicit and auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeVar


class TemporalFramePlanError(ValueError):
    """Raised when a temporal frame plan cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class TemporalFramePlan:
    """Requested film timing and the foundation-native generation frame count."""

    requested_frames: int
    generated_frames: int
    export_frames: int
    fps: float
    duration_seconds: float
    temporal_compression_ratio: int | None

    @property
    def requires_trim(self) -> bool:
        return self.generated_frames != self.export_frames


T = TypeVar("T")


def plan_temporal_frames(
    duration_seconds: float,
    fps: float,
    *,
    temporal_compression_ratio: int | None = None,
    max_generation_frames: int | None = None,
) -> TemporalFramePlan:
    """Plan exact shot timing without relying on a foundation's silent rounding.

    When ``temporal_compression_ratio`` is provided, ``generated_frames`` is the
    smallest count greater than or equal to the requested film frames satisfying
    ``(generated_frames - 1) % temporal_compression_ratio == 0``. CINEOS then
    exports exactly ``requested_frames`` after inference.
    """

    if not isinstance(duration_seconds, (int, float)) or isinstance(
        duration_seconds, bool
    ):
        raise TemporalFramePlanError("duration_seconds must be numeric")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool):
        raise TemporalFramePlanError("fps must be numeric")
    duration_seconds = float(duration_seconds)
    fps = float(fps)
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise TemporalFramePlanError("duration_seconds must be finite and positive")
    if not math.isfinite(fps) or fps <= 0:
        raise TemporalFramePlanError("fps must be finite and positive")

    requested_frames = max(1, round(duration_seconds * fps))
    generated_frames = requested_frames

    if temporal_compression_ratio is not None:
        if (
            isinstance(temporal_compression_ratio, bool)
            or not isinstance(temporal_compression_ratio, int)
            or temporal_compression_ratio <= 0
        ):
            raise TemporalFramePlanError(
                "temporal_compression_ratio must be a positive integer"
            )
        remainder = (requested_frames - 1) % temporal_compression_ratio
        if remainder:
            generated_frames += temporal_compression_ratio - remainder

    if max_generation_frames is not None:
        if (
            isinstance(max_generation_frames, bool)
            or not isinstance(max_generation_frames, int)
            or max_generation_frames <= 0
        ):
            raise TemporalFramePlanError(
                "max_generation_frames must be a positive integer"
            )
        if generated_frames > max_generation_frames:
            raise TemporalFramePlanError(
                "foundation-native frame plan exceeds max_generation_frames: "
                f"{generated_frames} > {max_generation_frames}"
            )

    return TemporalFramePlan(
        requested_frames=requested_frames,
        generated_frames=generated_frames,
        export_frames=requested_frames,
        fps=fps,
        duration_seconds=duration_seconds,
        temporal_compression_ratio=temporal_compression_ratio,
    )


def trim_generated_frames(frames: list[T], plan: TemporalFramePlan) -> list[T]:
    """Fail closed on foundation frame-count drift and return exact film frames."""

    actual = len(frames)
    if actual != plan.generated_frames:
        raise TemporalFramePlanError(
            "foundation returned an unexpected frame count: "
            f"expected {plan.generated_frames}, got {actual}"
        )
    return list(frames[: plan.export_frames])


__all__ = [
    "TemporalFramePlan",
    "TemporalFramePlanError",
    "plan_temporal_frames",
    "trim_generated_frames",
]
