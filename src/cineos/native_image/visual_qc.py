"""Provider-neutral multi-axis visual continuity quality control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


VISUAL_QC_AXES = (
    "face_identity",
    "body_shape",
    "wardrobe",
    "hair",
    "props",
    "environment",
    "lighting",
    "screen_direction",
)


@dataclass(frozen=True, slots=True)
class VisualContinuityObservation:
    """Normalized continuity scores extracted from one generated frame."""

    shot_id: str
    scores: Mapping[str, float]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.shot_id:
            raise ValueError("visual continuity observation requires a shot ID")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("visual continuity confidence must be between 0 and 1")
        unknown = set(self.scores) - set(VISUAL_QC_AXES)
        if unknown:
            raise ValueError(f"unknown visual QC axes: {sorted(unknown)}")
        if not self.scores:
            raise ValueError("visual continuity observation requires scores")
        if any(not 0.0 <= score <= 1.0 for score in self.scores.values()):
            raise ValueError("visual continuity scores must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class VisualQCReport:
    shot_id: str
    decision: str
    aggregate_score: float
    failed_axes: tuple[str, ...] = ()
    warning_axes: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.decision in {"accept", "warn"}

    @property
    def should_rerender(self) -> bool:
        return self.decision == "reject"


@dataclass(slots=True)
class VisualContinuityMemory:
    """Store only accepted visual continuity observations."""

    _accepted: dict[str, VisualContinuityObservation] = field(default_factory=dict)

    def latest(self, shot_id: str) -> VisualContinuityObservation | None:
        return self._accepted.get(shot_id)

    def accept(self, observation: VisualContinuityObservation) -> None:
        self._accepted[observation.shot_id] = observation


class MultiAxisVisualQCGate:
    """Aggregate visual continuity evidence into accept/warn/reject decisions."""

    def __init__(
        self,
        *,
        warning_score: float = 0.85,
        reject_score: float = 0.70,
        critical_axes: tuple[str, ...] = ("face_identity", "body_shape"),
    ) -> None:
        if not 0.0 <= reject_score <= warning_score <= 1.0:
            raise ValueError("visual QC thresholds must satisfy reject <= warning")
        unknown = set(critical_axes) - set(VISUAL_QC_AXES)
        if unknown:
            raise ValueError(f"unknown critical visual QC axes: {sorted(unknown)}")
        self.warning_score = warning_score
        self.reject_score = reject_score
        self.critical_axes = critical_axes

    def evaluate(self, observation: VisualContinuityObservation) -> VisualQCReport:
        failed = tuple(
            axis for axis, score in observation.scores.items() if score < self.reject_score
        )
        warnings = tuple(
            axis
            for axis, score in observation.scores.items()
            if self.reject_score <= score < self.warning_score
        )
        aggregate = sum(observation.scores.values()) / len(observation.scores)
        critical_failure = any(axis in self.critical_axes for axis in failed)
        reasons: list[str] = []

        if failed:
            reasons.append("one or more continuity axes fell below rejection threshold")
        if critical_failure:
            reasons.append("critical character continuity failure")
        if observation.confidence < 0.5:
            reasons.append("low visual continuity observation confidence")

        if critical_failure or aggregate < self.reject_score:
            decision = "reject"
        elif failed or warnings or aggregate < self.warning_score:
            decision = "warn"
        else:
            decision = "accept"

        return VisualQCReport(
            shot_id=observation.shot_id,
            decision=decision,
            aggregate_score=aggregate,
            failed_axes=failed,
            warning_axes=warnings,
            reasons=tuple(reasons),
        )


def build_rerender_directives(report: VisualQCReport) -> tuple[str, ...]:
    """Translate QC failures into deterministic renderer-neutral corrections."""
    directives = [f"preserve {axis.replace('_', ' ')}" for axis in report.failed_axes]
    if report.should_rerender:
        directives.append("rerender shot from last accepted continuity state")
    return tuple(directives)
