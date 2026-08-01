"""Renderer-independent validator interfaces and result values."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class ValidationStatus(StrEnum):
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    FAIL = "fail"
    UNSUPPORTED = "unsupported"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


@dataclass(slots=True)
class ValidationResult:
    category: str
    status: ValidationStatus
    score: float | None
    checks: dict[str, float | bool | str | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ValidatorBackend(Protocol):
    """Optional model adapter boundary; implementations never identify people."""

    backend_id: str

    def score(self, capability: str, expected: Any, frames: list[Path]) -> float | None:
        """Return normalized consistency, or ``None`` when unsupported."""

    def temporal_metrics(self, frames: list[Path]) -> dict[str, float] | None:
        """Return normalized drift/instability values where zero is best."""


class BaseValidator(ABC):
    category: str

    @abstractmethod
    def validate(self, context: Any) -> ValidationResult:
        """Validate one category against extracted render frames."""


class FakeValidatorBackend:
    """Deterministic backend for tests and pipeline integration exercises."""

    backend_id = "fake"

    def __init__(
        self,
        scores: dict[str, float | None] | None = None,
        temporal: dict[str, float] | None = None,
    ) -> None:
        self.scores = dict(scores or {})
        self.temporal = dict(temporal or {})

    def score(self, capability: str, expected: Any, frames: list[Path]) -> float | None:
        del expected, frames
        return self.scores.get(capability, 1.0)

    def temporal_metrics(self, frames: list[Path]) -> dict[str, float] | None:
        del frames
        return dict(self.temporal)
