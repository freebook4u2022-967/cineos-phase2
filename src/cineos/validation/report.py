"""Structured validation report."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .base import ValidationResult, ValidationStatus


@dataclass(slots=True)
class ValidationReport:
    shot_id: str
    scene_id: str
    renderer_id: str
    overall_status: ValidationStatus
    overall_score: float | None
    results: list[ValidationResult]
    report_uuid: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    rerender_recommendation: str | None = None
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def report_id(self) -> str:
        """Compatibility alias for the report UUID."""
        return self.report_uuid

    @property
    def validation_results(self) -> list[ValidationResult]:
        """Descriptive alias for the individual category results."""
        return self.results

    @property
    def should_rerender(self) -> bool:
        return self.overall_status is ValidationStatus.FAIL
