"""Plan-aware duration integrity checks for assembled CINEOS movies.

A movie can contain decodable, visually plausible frames and still be incomplete if
assembly silently truncates a shot, repeats footage, or stops early. This module
keeps final acceptance tied to authored duration evidence by comparing the planned
shot duration with the duration reported by the encoded media container.

FFprobe is used only to inspect the completed artifact; it is not a rendering or
visual-generation dependency. The pure comparison layer is dependency-free so its
policy can be regression-tested in CI.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DurationProbeError(RuntimeError):
    """Raised when encoded movie duration cannot be measured safely."""


@dataclass(frozen=True, slots=True)
class DurationIntegrityPolicy:
    """Versionable tolerance for final-film duration drift."""

    absolute_tolerance_seconds: float = 0.25
    relative_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if self.absolute_tolerance_seconds < 0.0:
            raise ValueError("absolute_tolerance_seconds must be non-negative")
        if not 0.0 <= self.relative_tolerance <= 1.0:
            raise ValueError("relative_tolerance must be in [0, 1]")

    def allowed_error(self, planned_seconds: float) -> float:
        if planned_seconds <= 0.0:
            raise ValueError("planned_seconds must be positive")
        return max(
            self.absolute_tolerance_seconds,
            planned_seconds * self.relative_tolerance,
        )


@dataclass(frozen=True, slots=True)
class DurationIntegrityReport:
    """Auditable duration evidence for a completed film artifact."""

    planned_seconds: float
    measured_seconds: float
    delta_seconds: float
    allowed_error_seconds: float
    decision: str
    directives: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"


def planned_duration(plan: Sequence[Any]) -> float:
    """Return authored movie duration, failing on malformed shot plans."""

    if not plan:
        raise ValueError("duration integrity requires a non-empty shot plan")
    total = 0.0
    for shot in plan:
        duration = float(getattr(shot, "duration", 0.0))
        if duration <= 0.0:
            raise ValueError("planned shot duration must be positive")
        total += duration
    return total


def evaluate_duration_integrity(
    planned_seconds: float,
    measured_seconds: float,
    policy: DurationIntegrityPolicy | None = None,
) -> DurationIntegrityReport:
    """Compare authored and encoded duration and fail closed on excess drift."""

    if planned_seconds <= 0.0:
        raise ValueError("planned_seconds must be positive")
    if measured_seconds <= 0.0:
        raise ValueError("measured_seconds must be positive")
    active_policy = policy or DurationIntegrityPolicy()
    delta = measured_seconds - planned_seconds
    allowed = active_policy.allowed_error(planned_seconds)
    if abs(delta) <= allowed:
        return DurationIntegrityReport(
            planned_seconds=planned_seconds,
            measured_seconds=measured_seconds,
            delta_seconds=delta,
            allowed_error_seconds=allowed,
            decision="accept",
        )

    direction = "shorter" if delta < 0.0 else "longer"
    return DurationIntegrityReport(
        planned_seconds=planned_seconds,
        measured_seconds=measured_seconds,
        delta_seconds=delta,
        allowed_error_seconds=allowed,
        decision="reject",
        directives=(
            "assembled movie duration is "
            f"{abs(delta):.3f}s {direction} than the authored shot plan; "
            "rebuild assembly and verify shot inclusion/order before release",
        ),
    )


@dataclass(slots=True)
class FFprobeDurationIntegrityGate:
    """Measure a completed movie and compare it with the authored shot plan."""

    policy: DurationIntegrityPolicy | None = None
    ffprobe_binary: str = "ffprobe"

    def evaluate(
        self,
        movie_path: str | Path,
        plan: Sequence[Any],
    ) -> DurationIntegrityReport:
        source = Path(movie_path)
        if not source.is_file() or source.stat().st_size <= 0:
            raise DurationProbeError(f"final movie is missing or empty: {source}")

        ffprobe = shutil.which(self.ffprobe_binary)
        if ffprobe is None:
            raise DurationProbeError(
                f"FFprobe inspector {self.ffprobe_binary!r} is not available"
            )

        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(source),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise DurationProbeError(
                "FFprobe failed to inspect final movie duration: "
                + completed.stderr.strip()
            )
        try:
            measured = float(completed.stdout.strip())
        except ValueError as exc:
            raise DurationProbeError(
                "FFprobe returned a non-numeric final movie duration"
            ) from exc
        if measured <= 0.0:
            raise DurationProbeError("FFprobe returned a non-positive movie duration")

        return evaluate_duration_integrity(
            planned_duration(plan),
            measured,
            self.policy,
        )


__all__ = [
    "DurationIntegrityPolicy",
    "DurationIntegrityReport",
    "DurationProbeError",
    "FFprobeDurationIntegrityGate",
    "evaluate_duration_integrity",
    "planned_duration",
]
