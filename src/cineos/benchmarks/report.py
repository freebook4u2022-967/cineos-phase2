"""Benchmark execution reports."""

from __future__ import annotations

from dataclasses import dataclass, field

from .metrics import Metric


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: str
    passed: bool
    metrics: tuple[Metric, ...]
    warnings: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    deterministic_hash: str = ""


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    suite_id: str
    suite_version: str
    suite_hash: str
    renderer_profile: str
    hardware_profile: str
    results: tuple[CaseResult, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.results)
