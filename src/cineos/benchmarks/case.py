"""Benchmark case contract."""

from __future__ import annotations

from dataclasses import dataclass, field

from .exceptions import BenchmarkError


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    title: str
    purpose: str
    project_fixture: str
    required_assets: tuple[str, ...] = ()
    renderer_requirements: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ("report.json",)
    maximum_runtime: float = 30.0
    hardware_requirements: dict[str, object] = field(default_factory=dict)
    deterministic_seed: int = 0
    validation_thresholds: dict[str, float] = field(default_factory=dict)
    mandatory: bool = True
    slow: bool = False

    def __post_init__(self) -> None:
        if not self.case_id or not self.title or not self.purpose:
            raise BenchmarkError("case ID, title, and purpose are required")
        if not self.project_fixture or self.maximum_runtime <= 0:
            raise BenchmarkError("a fixture and positive maximum runtime are required")
        if any(not 0 <= value <= 1 for value in self.validation_thresholds.values()):
            raise BenchmarkError("validation thresholds must be between zero and one")
