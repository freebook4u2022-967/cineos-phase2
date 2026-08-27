"""Dependency-free temporal stability metrics for scene-boundary windows.

Final-film boundary QC samples several decoded frames on both sides of an authored
edit. Cross-boundary MAD alone can miss a transient flicker when the outgoing and
incoming windows contain the same defect at corresponding offsets. This module
measures *within-side* temporal stability so corrupt flashes, exposure jumps, or
single-frame generation failures cannot hide behind an otherwise plausible edit.

The metric is intentionally independent from FFmpeg and media containers. Production
adapters supply real decoded grayscale frames; CI can exercise the exact decision
logic with deterministic byte evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoundaryWindowStabilityPolicy:
    """Versionable thresholds for temporal instability beside an edit."""

    warn_peak_mad: float = 24.0
    reject_peak_mad: float = 48.0

    def __post_init__(self) -> None:
        if self.warn_peak_mad < 0.0 or self.reject_peak_mad < 0.0:
            raise ValueError("boundary-window MAD thresholds must be non-negative")
        if self.warn_peak_mad > self.reject_peak_mad:
            raise ValueError("warn_peak_mad cannot exceed reject_peak_mad")


@dataclass(frozen=True, slots=True)
class BoundaryWindowStabilityReport:
    """Auditable within-side motion evidence surrounding one scene boundary."""

    sample_count_per_side: int
    outgoing_mean_mad: float
    incoming_mean_mad: float
    outgoing_peak_mad: float
    incoming_peak_mad: float
    decision: str
    directives: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.decision in {"accept", "warn"}


def _frame_mad(left: bytes, right: bytes) -> float:
    if not left or len(left) != len(right):
        raise ValueError("boundary-window frames must have identical non-zero sizes")
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / len(left)


def _validate_window(frames: Sequence[bytes], *, label: str) -> int:
    if len(frames) < 2:
        raise ValueError(f"{label} boundary window requires at least two frames")
    frame_size = len(frames[0])
    if frame_size <= 0:
        raise ValueError(f"{label} boundary window frames cannot be empty")
    if any(len(frame) != frame_size for frame in frames):
        raise ValueError(
            f"{label} boundary window frames must have identical non-zero sizes"
        )
    return frame_size


def _adjacent_mads(frames: Sequence[bytes]) -> tuple[float, ...]:
    return tuple(
        _frame_mad(left, right)
        for left, right in zip(frames, frames[1:], strict=False)
    )


def evaluate_boundary_window_stability(
    outgoing_frames: Sequence[bytes],
    incoming_frames: Sequence[bytes],
    policy: BoundaryWindowStabilityPolicy | None = None,
) -> BoundaryWindowStabilityReport:
    """Detect transient instability immediately before or after an authored edit.

    Both windows must contain the same number of equally sized decoded frames.
    The evaluator deliberately does not compare the two sides of the edit; that is
    the responsibility of edit-aware scene-boundary QC. Instead it asks whether
    either side is internally unstable enough to indicate a flash/corrupt frame or
    a short-lived generation discontinuity that cross-boundary averaging may hide.
    """

    outgoing_size = _validate_window(outgoing_frames, label="outgoing")
    incoming_size = _validate_window(incoming_frames, label="incoming")
    if len(outgoing_frames) != len(incoming_frames):
        raise ValueError("boundary windows must contain the same number of frames")
    if outgoing_size != incoming_size:
        raise ValueError("outgoing and incoming boundary frames must have equal sizes")

    active_policy = policy or BoundaryWindowStabilityPolicy()
    outgoing_mads = _adjacent_mads(outgoing_frames)
    incoming_mads = _adjacent_mads(incoming_frames)
    outgoing_peak = max(outgoing_mads)
    incoming_peak = max(incoming_mads)
    outgoing_mean = sum(outgoing_mads) / len(outgoing_mads)
    incoming_mean = sum(incoming_mads) / len(incoming_mads)
    peak = max(outgoing_peak, incoming_peak)

    directives: list[str] = []
    if peak >= active_policy.reject_peak_mad:
        decision = "reject"
        directives.append(
            "rerender transient temporal instability inside the scene-boundary window"
        )
    elif peak >= active_policy.warn_peak_mad:
        decision = "warn"
        directives.append(
            "review scene-boundary window for transient flicker or exposure drift"
        )
    else:
        decision = "accept"

    return BoundaryWindowStabilityReport(
        sample_count_per_side=len(outgoing_frames),
        outgoing_mean_mad=outgoing_mean,
        incoming_mean_mad=incoming_mean,
        outgoing_peak_mad=outgoing_peak,
        incoming_peak_mad=incoming_peak,
        decision=decision,
        directives=tuple(directives),
    )


__all__ = [
    "BoundaryWindowStabilityPolicy",
    "BoundaryWindowStabilityReport",
    "evaluate_boundary_window_stability",
]
