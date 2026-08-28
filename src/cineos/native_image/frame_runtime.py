"""Executable CINEOS native frame-generation prototype runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .backend import NativeImageResearchBackend, NativeImageResearchResult
from .conditioning import NativeImageConditioningPlan
from .rerender import AutomaticRerenderController, RerenderDecision, correction_payload
from .temporal_identity import IdentityObservation, TemporalIdentityMemory
from .visual_qc import VisualContinuityObservation


class NativeFrameObserver(Protocol):
    """Extract provider-neutral QC observations from a generated native frame."""

    def observe_identity(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> tuple[IdentityObservation, ...]: ...

    def observe_visual_continuity(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> VisualContinuityObservation: ...


class CorrectionAwareNativeImageModel(Protocol):
    """Optional extension for models that can consume rerender corrections."""

    def apply_corrections(self, payload: dict[str, object]) -> None: ...


@dataclass(frozen=True, slots=True)
class NativeFrameAttempt:
    attempt: int
    result: NativeImageResearchResult
    decision: RerenderDecision


@dataclass(frozen=True, slots=True)
class NativeFrameGenerationResult:
    shot_id: str
    accepted: bool
    final_decision: str
    attempts: tuple[NativeFrameAttempt, ...]
    image: Any | None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


class NativeFrameRuntime:
    """Generate one frame, run QC, and automatically rerender when required.

    Observers may expose an ``accept_frame(result, plan)`` method when they own
    transactional continuity state derived from generated pixels. Observers with
    mutable commit state should additionally expose ``checkpoint_state()`` and
    ``restore_state(checkpoint)``. Identity memory and observer-owned continuity
    state are then committed as one acceptance transaction: if either commit path
    fails, both stores are restored to their pre-evaluation checkpoints so a
    partially accepted frame can never become future context.
    """

    def __init__(
        self,
        backend: NativeImageResearchBackend,
        observer: NativeFrameObserver,
        *,
        controller: AutomaticRerenderController | None = None,
        identity_memory: TemporalIdentityMemory | None = None,
    ) -> None:
        self.backend = backend
        self.observer = observer
        self.controller = controller or AutomaticRerenderController()
        self.identity_memory = identity_memory or TemporalIdentityMemory()

    def generate(
        self, plan: NativeImageConditioningPlan
    ) -> NativeFrameGenerationResult:
        if not isinstance(plan, NativeImageConditioningPlan):
            raise TypeError("plan must be a NativeImageConditioningPlan")

        attempts: list[NativeFrameAttempt] = []
        for attempt in range(1, self.controller.max_attempts + 1):
            result = self.backend.render(plan)
            identities = self.observer.observe_identity(result, plan)
            visual = self.observer.observe_visual_continuity(result, plan)
            identity_checkpoint = self.identity_memory.checkpoint()
            observer_checkpoint: object | None = None
            checkpoint_state = getattr(self.observer, "checkpoint_state", None)
            restore_state = getattr(self.observer, "restore_state", None)
            if callable(checkpoint_state):
                if not callable(restore_state):
                    raise TypeError(
                        "observer checkpoint_state requires matching restore_state"
                    )
                observer_checkpoint = checkpoint_state()

            decision = self.controller.evaluate(
                self.identity_memory,
                identities,
                visual,
                attempt=attempt,
            )
            attempts.append(
                NativeFrameAttempt(
                    attempt=attempt,
                    result=result,
                    decision=decision,
                )
            )

            if decision.accepted:
                accept_frame = getattr(self.observer, "accept_frame", None)
                if callable(accept_frame):
                    try:
                        accept_frame(result, plan)
                    except Exception:
                        self.identity_memory.restore(identity_checkpoint)
                        if observer_checkpoint is not None:
                            restore_state(observer_checkpoint)
                        raise
                return NativeFrameGenerationResult(
                    shot_id=plan.shot_id,
                    accepted=True,
                    final_decision=decision.decision,
                    attempts=tuple(attempts),
                    image=result.image,
                )
            if not decision.should_rerender:
                break

            payload = dict(correction_payload(decision))
            model = self.backend.model
            apply_corrections = getattr(model, "apply_corrections", None)
            if callable(apply_corrections):
                apply_corrections(payload)
            plan.metadata["rerender_corrections"] = payload
            plan.metadata["rerender_attempt"] = attempt + 1
            plan.refresh_hash()

        return NativeFrameGenerationResult(
            shot_id=plan.shot_id,
            accepted=False,
            final_decision=attempts[-1].decision.decision,
            attempts=tuple(attempts),
            image=None,
        )
