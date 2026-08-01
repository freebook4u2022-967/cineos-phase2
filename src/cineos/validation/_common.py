from __future__ import annotations

from typing import Any

from .base import ValidationResult, ValidationStatus


def value(source: Any, name: str, default: Any = None) -> Any:
    return (
        source.get(name, default)
        if isinstance(source, dict)
        else getattr(source, name, default)
    )


def scored_result(
    category: str, scores: dict[str, float | None], threshold: float
) -> ValidationResult:
    supported = [score for score in scores.values() if score is not None]
    if not supported:
        return ValidationResult(
            category, ValidationStatus.UNSUPPORTED, None, checks=scores
        )
    score = sum(supported) / len(supported)
    unsupported = [name for name, item in scores.items() if item is None]
    failed_checks = [
        name for name, item in scores.items() if item is not None and item < threshold
    ]
    if score < threshold or failed_checks:
        status = ValidationStatus.FAIL
        failures = [
            f"{category} checks below {threshold:.3f}: " + ", ".join(failed_checks)
        ]
        warnings: list[str] = []
    elif unsupported:
        status = ValidationStatus.PASS_WITH_WARNINGS
        failures = []
        warnings = ["unsupported checks: " + ", ".join(unsupported)]
    else:
        status, failures, warnings = ValidationStatus.PASS, [], []
    return ValidationResult(category, status, score, scores, warnings, failures)
