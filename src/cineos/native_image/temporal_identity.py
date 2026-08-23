"""Temporal character identity memory and provider-neutral visual QC."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .conditioning import NativeImageConditioningPlan


class IdentityObservationError(ValueError):
    """Raised when an identity observation cannot be compared safely."""


@dataclass(frozen=True, slots=True)
class IdentityObservation:
    """Model-neutral identity evidence extracted from one generated shot."""

    character_uuid: str
    shot_id: str
    embedding: tuple[float, ...]
    approved_reference_ids: tuple[str, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.character_uuid or not self.shot_id:
            raise IdentityObservationError(
                "identity observation requires character and shot IDs"
            )
        if not self.embedding:
            raise IdentityObservationError("identity observation requires an embedding")
        if any(not math.isfinite(value) for value in self.embedding):
            raise IdentityObservationError("identity embedding values must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise IdentityObservationError(
                "identity confidence must be between 0 and 1"
            )


@dataclass(frozen=True, slots=True)
class IdentityQCReport:
    character_uuid: str
    shot_id: str
    baseline_shot_id: str | None
    similarity: float
    drift_score: float
    confidence: float
    decision: str
    reasons: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.decision in {"baseline", "accept", "warn"}

    @property
    def should_rerender(self) -> bool:
        return self.decision == "reject"


@dataclass(slots=True)
class TemporalIdentityMemory:
    """Carry the last accepted identity state for each character across shots."""

    _accepted: dict[str, IdentityObservation] = field(default_factory=dict)
    _history: dict[str, list[IdentityObservation]] = field(default_factory=dict)

    def latest(self, character_uuid: str) -> IdentityObservation | None:
        return self._accepted.get(character_uuid)

    def history(self, character_uuid: str) -> tuple[IdentityObservation, ...]:
        return tuple(self._history.get(character_uuid, ()))

    def accept(self, observation: IdentityObservation) -> None:
        self._accepted[observation.character_uuid] = observation
        self._history.setdefault(observation.character_uuid, []).append(observation)


def _cosine_similarity(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    if len(first) != len(second):
        raise IdentityObservationError("identity embeddings must have equal dimensions")
    dot = sum(left * right for left, right in zip(first, second, strict=True))
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise IdentityObservationError(
            "identity embeddings must have non-zero magnitude"
        )
    cosine = dot / (first_norm * second_norm)
    return max(-1.0, min(1.0, cosine))


class IdentityVisualQCGate:
    """Accept, warn, or reject generated identity evidence before carry-forward."""

    def __init__(
        self,
        *,
        warning_drift: float = 0.15,
        reject_drift: float = 0.30,
    ) -> None:
        if not 0.0 <= warning_drift <= reject_drift <= 1.0:
            raise ValueError(
                "identity drift thresholds must satisfy 0 <= warning <= reject <= 1"
            )
        self.warning_drift = warning_drift
        self.reject_drift = reject_drift

    def evaluate(
        self,
        memory: TemporalIdentityMemory,
        observation: IdentityObservation,
        *,
        update_memory: bool = True,
    ) -> IdentityQCReport:
        baseline = memory.latest(observation.character_uuid)
        if baseline is None:
            report = IdentityQCReport(
                character_uuid=observation.character_uuid,
                shot_id=observation.shot_id,
                baseline_shot_id=None,
                similarity=1.0,
                drift_score=0.0,
                confidence=observation.confidence,
                decision="baseline",
            )
            if update_memory:
                memory.accept(observation)
            return report

        cosine = _cosine_similarity(baseline.embedding, observation.embedding)
        similarity = (cosine + 1.0) / 2.0
        drift = 1.0 - similarity
        reasons: list[str] = []
        if observation.confidence < 0.5:
            reasons.append("low identity observation confidence")

        if drift >= self.reject_drift:
            decision = "reject"
            reasons.append("identity drift exceeds rejection threshold")
        elif drift >= self.warning_drift:
            decision = "warn"
            reasons.append("identity drift exceeds warning threshold")
        else:
            decision = "accept"

        report = IdentityQCReport(
            character_uuid=observation.character_uuid,
            shot_id=observation.shot_id,
            baseline_shot_id=baseline.shot_id,
            similarity=similarity,
            drift_score=drift,
            confidence=observation.confidence,
            decision=decision,
            reasons=tuple(reasons),
        )
        if report.accepted and update_memory:
            memory.accept(observation)
        return report


def apply_temporal_identity_memory(
    plan: NativeImageConditioningPlan,
    memory: TemporalIdentityMemory,
) -> NativeImageConditioningPlan:
    """Attach the previous accepted identity state to the next shot's conditioning."""
    if not isinstance(plan, NativeImageConditioningPlan):
        raise TypeError("plan must be a NativeImageConditioningPlan")

    contexts: list[dict[str, Any]] = []
    for identity in plan.identity_tokens:
        character_uuid = str(identity.get("character_uuid", ""))
        previous = memory.latest(character_uuid)
        if previous is None:
            continue
        contexts.append(
            {
                "character_uuid": character_uuid,
                "previous_shot_id": previous.shot_id,
                "previous_identity_embedding": list(previous.embedding),
                "approved_reference_ids": list(previous.approved_reference_ids),
                "confidence": previous.confidence,
            }
        )

    plan.metadata["temporal_identity_context"] = contexts
    plan.metadata["temporal_identity_memory_enabled"] = True
    plan.refresh_hash()
    return plan
