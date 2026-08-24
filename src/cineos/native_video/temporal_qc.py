"""Transactional QC for CINEOS native temporal frame generation.

The gate operates on immutable temporal candidates and never mutates sequence
state. Callers commit a candidate only after the gate accepts it, preserving the
last known-good continuity state across retries and process resumes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .temporal_model import TemporalFrameOutput, TemporalSequenceState


@dataclass(frozen=True, slots=True)
class TemporalQCPolicy:
    """Versionable thresholds for frame-to-frame latent continuity."""

    warn_delta: float = 0.35
    reject_delta: float = 0.75

    def __post_init__(self) -> None:
        if self.warn_delta < 0 or self.reject_delta < 0:
            raise ValueError("temporal QC thresholds must be non-negative")
        if self.warn_delta > self.reject_delta:
            raise ValueError("warn_delta cannot exceed reject_delta")


@dataclass(frozen=True, slots=True)
class TemporalQCReport:
    """Decision for one proposed temporal frame candidate."""

    shot_id: str
    frame_index: int
    decision: str
    continuity_delta: float
    threshold: float
    directives: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.decision in {"accept", "warn"}

    @property
    def should_retry(self) -> bool:
        return self.decision == "retry"


class TemporalContinuityGate:
    """Reject excessive temporal jumps before recurrent state is committed."""

    def __init__(self, policy: TemporalQCPolicy | None = None) -> None:
        self.policy = policy or TemporalQCPolicy()

    def evaluate(
        self,
        candidate: TemporalFrameOutput,
        state: TemporalSequenceState,
    ) -> TemporalQCReport:
        if candidate.shot_id != state.shot_id:
            raise ValueError("candidate and temporal state must belong to the same shot")
        expected_index = state.last_frame_index + 1
        if candidate.frame_index != expected_index:
            raise ValueError(
                f"expected candidate frame_index {expected_index}, got {candidate.frame_index}"
            )
        if candidate.continuity_delta < 0:
            raise ValueError("continuity_delta must be non-negative")

        delta = candidate.continuity_delta
        if delta >= self.policy.reject_delta:
            return TemporalQCReport(
                shot_id=candidate.shot_id,
                frame_index=candidate.frame_index,
                decision="retry",
                continuity_delta=delta,
                threshold=self.policy.reject_delta,
                directives=(
                    "reduce temporal latent jump while preserving approved identity",
                    "preserve last accepted camera, scene and wardrobe continuity state",
                    "do not commit rejected recurrent state",
                ),
            )
        if delta >= self.policy.warn_delta:
            return TemporalQCReport(
                shot_id=candidate.shot_id,
                frame_index=candidate.frame_index,
                decision="warn",
                continuity_delta=delta,
                threshold=self.policy.warn_delta,
                directives=("monitor temporal drift on the next accepted frame",),
            )
        return TemporalQCReport(
            shot_id=candidate.shot_id,
            frame_index=candidate.frame_index,
            decision="accept",
            continuity_delta=delta,
            threshold=self.policy.warn_delta,
        )
