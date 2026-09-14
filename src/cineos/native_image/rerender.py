"""Automatic rerender orchestration for native visual continuity QC."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .temporal_identity import (
    IdentityObservation,
    IdentityQCReport,
    IdentityVisualQCGate,
    TemporalIdentityMemory,
)
from .visual_qc import (
    MultiAxisVisualQCGate,
    VisualContinuityObservation,
    VisualQCReport,
    build_rerender_directives,
)


@dataclass(frozen=True, slots=True)
class RerenderDecision:
    shot_id: str
    decision: str
    attempt: int
    max_attempts: int
    identity_reports: tuple[IdentityQCReport, ...]
    visual_report: VisualQCReport
    directives: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.decision in {"accept", "warn"}

    @property
    def should_rerender(self) -> bool:
        return self.decision == "rerender"


class AutomaticRerenderController:
    """Gate generated shots before any continuity state is carried forward."""

    def __init__(
        self,
        *,
        identity_gate: IdentityVisualQCGate | None = None,
        visual_gate: MultiAxisVisualQCGate | None = None,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.identity_gate = identity_gate or IdentityVisualQCGate()
        self.visual_gate = visual_gate or MultiAxisVisualQCGate()
        self.max_attempts = max_attempts

    def evaluate(
        self,
        memory: TemporalIdentityMemory,
        identity_observations: tuple[IdentityObservation, ...],
        visual_observation: VisualContinuityObservation,
        *,
        attempt: int = 1,
    ) -> RerenderDecision:
        if not 1 <= attempt <= self.max_attempts:
            raise ValueError("attempt must be within configured rerender budget")
        mismatched = any(
            item.shot_id != visual_observation.shot_id for item in identity_observations
        )
        if mismatched:
            raise ValueError(
                "identity and visual observations must belong to the same shot"
            )

        identity_reports = tuple(
            self.identity_gate.evaluate(memory, item, update_memory=False)
            for item in identity_observations
        )
        visual_report = self.visual_gate.evaluate(visual_observation)
        rejected_identity = any(report.should_rerender for report in identity_reports)
        rejected = rejected_identity or visual_report.should_rerender
        warnings = visual_report.decision == "warn" or any(
            report.decision == "warn" for report in identity_reports
        )

        directives = list(build_rerender_directives(visual_report))
        for report in identity_reports:
            if report.should_rerender:
                directives.append(
                    f"restore approved identity for {report.character_uuid}"
                )
        directives = list(dict.fromkeys(directives))

        if rejected and attempt < self.max_attempts:
            decision = "rerender"
        elif rejected:
            decision = "reject"
            directives.append(
                "rerender budget exhausted; require human or higher-tier review"
            )
        elif warnings:
            decision = "warn"
        else:
            decision = "accept"

        if decision in {"accept", "warn"}:
            for observation in identity_observations:
                memory.accept(observation)

        return RerenderDecision(
            shot_id=visual_observation.shot_id,
            decision=decision,
            attempt=attempt,
            max_attempts=self.max_attempts,
            identity_reports=identity_reports,
            visual_report=visual_report,
            directives=tuple(directives),
        )


def correction_payload(decision: RerenderDecision) -> Mapping[str, object]:
    """Return renderer-neutral corrections for the next generation attempt."""
    return {
        "shot_id": decision.shot_id,
        "attempt": decision.attempt + 1,
        "directives": list(decision.directives),
        "preserve_last_accepted_identity": True,
        "do_not_commit_rejected_state": True,
    }
