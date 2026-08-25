"""Production-safe temporal generation runtime for CINEOS native video.

The runtime owns the propose -> QC -> retry -> commit transaction. Rejected
candidates never advance recurrent state, so a failed frame cannot poison later
continuity or resumable checkpoints. Retry adaptation is explicit and versionable;
it modifies only the requested frame inputs while preserving the last accepted
sequence state. Versioned observability events are emitted fail-open so telemetry
failures never corrupt or halt native rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from cineos.native_image.tensor_model import Tensor

from .observability import (
    NullTemporalObserver,
    TemporalObserver,
    TemporalRuntimeEvent,
)
from .temporal_model import (
    NativeTemporalModel,
    TemporalFrameInput,
    TemporalFrameOutput,
    TemporalSequenceState,
)
from .temporal_qc import TemporalContinuityGate, TemporalQCReport


class TemporalRetryPolicy(Protocol):
    """Adapt a rejected frame request without mutating accepted sequence state."""

    def adapt(
        self,
        frame: TemporalFrameInput,
        report: TemporalQCReport,
        *,
        attempt: int,
    ) -> TemporalFrameInput: ...


@dataclass(frozen=True, slots=True)
class MotionDampingRetryPolicy:
    """Conservative retry policy that reduces requested motion after drift.

    This is intentionally a policy, not hidden renderer behavior. Future learned
    retry controllers can implement :class:`TemporalRetryPolicy` while preserving
    the same transactional state guarantees.
    """

    damping: float = 0.5
    minimum_scale: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.damping < 1.0:
            raise ValueError("damping must be between 0 and 1")
        if not 0.0 < self.minimum_scale <= 1.0:
            raise ValueError("minimum_scale must be in (0, 1]")

    def adapt(
        self,
        frame: TemporalFrameInput,
        report: TemporalQCReport,
        *,
        attempt: int,
    ) -> TemporalFrameInput:
        report_matches_frame = (
            report.shot_id == frame.shot_id and report.frame_index == frame.frame_index
        )
        if not report_matches_frame:
            raise ValueError("retry report must describe the rejected frame")
        if attempt <= 0:
            raise ValueError("retry attempt must be positive")

        scale = max(self.minimum_scale, self.damping**attempt)
        motion = Tensor(
            tuple(value * scale for value in frame.motion.values),
            frame.motion.shape,
            frame.motion.device,
        )
        return TemporalFrameInput(
            shot_id=frame.shot_id,
            frame_index=frame.frame_index,
            identity=frame.identity,
            scene=frame.scene,
            motion=motion,
        )


@dataclass(frozen=True, slots=True)
class TemporalGenerationResult:
    """Auditable outcome for one requested frame."""

    candidate: TemporalFrameOutput
    report: TemporalQCReport
    attempts: int


class TemporalGenerationError(RuntimeError):
    """Raised when a frame cannot pass temporal QC within the retry budget."""

    def __init__(self, report: TemporalQCReport, attempts: int) -> None:
        self.report = report
        self.attempts = attempts
        super().__init__(
            f"temporal frame {report.shot_id}:{report.frame_index} failed QC "
            f"after {attempts} attempts"
        )


@dataclass(slots=True)
class NativeTemporalRuntime:
    """Execute native temporal generation with transactional QC and retries."""

    model: NativeTemporalModel
    gate: TemporalContinuityGate
    retry_policy: TemporalRetryPolicy
    observer: TemporalObserver = field(default_factory=NullTemporalObserver)
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    @classmethod
    def default(
        cls,
        model: NativeTemporalModel | None = None,
        gate: TemporalContinuityGate | None = None,
        *,
        observer: TemporalObserver | None = None,
        max_retries: int = 2,
    ) -> NativeTemporalRuntime:
        return cls(
            model=model or NativeTemporalModel.initialized(),
            gate=gate or TemporalContinuityGate(),
            retry_policy=MotionDampingRetryPolicy(),
            observer=observer or NullTemporalObserver(),
            max_retries=max_retries,
        )

    def _record(
        self,
        report: TemporalQCReport,
        *,
        attempt: int,
        state: TemporalSequenceState,
    ) -> None:
        """Emit telemetry without allowing an observer failure to stop rendering."""
        event = TemporalRuntimeEvent(
            event_type=(
                "candidate_accepted" if report.accepted else "candidate_rejected"
            ),
            shot_id=report.shot_id,
            frame_index=report.frame_index,
            attempt=attempt,
            decision=report.decision,
            continuity_delta=report.continuity_delta,
            threshold=report.threshold,
        )
        try:
            self.observer.record(event)
        except Exception:
            state.metadata["temporal_observer_errors"] = (
                int(state.metadata.get("temporal_observer_errors", 0)) + 1
            )

    def generate_frame(
        self,
        frame: TemporalFrameInput,
        state: TemporalSequenceState,
    ) -> TemporalGenerationResult:
        """Generate one accepted frame or fail without corrupting sequence state."""
        request = frame
        attempts = 0

        while True:
            attempts += 1
            candidate = self.model.propose(request, state)
            report = self.gate.evaluate(candidate, state)
            self._record(report, attempt=attempts, state=state)

            if report.accepted:
                self.model.commit(candidate, state)
                state.metadata["temporal_attempts"] = (
                    int(state.metadata.get("temporal_attempts", 0)) + attempts
                )
                state.metadata["temporal_retries"] = int(
                    state.metadata.get("temporal_retries", 0)
                ) + (attempts - 1)
                return TemporalGenerationResult(candidate, report, attempts)

            retries_used = attempts - 1
            if retries_used >= self.max_retries:
                state.metadata["temporal_failed_candidates"] = (
                    int(state.metadata.get("temporal_failed_candidates", 0)) + attempts
                )
                raise TemporalGenerationError(report, attempts)

            request = self.retry_policy.adapt(
                frame,
                report,
                attempt=attempts,
            )
