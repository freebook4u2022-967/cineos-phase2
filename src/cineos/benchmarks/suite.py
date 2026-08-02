"""Versioned benchmark suite and built-in Alpha catalog."""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field

from .case import BenchmarkCase
from .metrics import METRIC_NAMES

CASE_TITLES = (
    "Single-character close-up",
    "Two-character dialogue",
    "Walking scene",
    "Emotional reaction",
    "Prop interaction",
    "Vehicle interaction",
    "Rain and weather",
    "Night lighting",
    "Camera movement",
    "Continuity across multiple shots",
    "Lip-sync dialogue",
    "Complete short-film build",
    "Post-production delivery",
)


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    suite_id: str
    suite_version: str
    cases: tuple[BenchmarkCase, ...]
    target_platform: str = "portable"
    renderer_profile: str = "deterministic-preview"
    metric_definitions: tuple[str, ...] = METRIC_NAMES
    baseline_id: str | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        uuid.UUID(self.suite_id)
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("benchmark case IDs must be unique")

    @property
    def content_hash(self) -> str:
        value = asdict(self)
        value.pop("baseline_id", None)
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


def alpha_suite() -> BenchmarkSuite:
    """Return the stable, CPU-only Alpha suite definition."""
    cases = tuple(
        BenchmarkCase(
            f"alpha-{index:02d}",
            title,
            f"Validate {title.lower()} workflow.",
            f"benchmarks/projects/alpha-{index:02d}.json",
            mandatory=index in {1, 2, 9, 10, 12, 13},
            slow=index in {12, 13},
        )
        for index, title in enumerate(CASE_TITLES, 1)
    )
    return BenchmarkSuite(
        "57d6ad14-86d9-5ea7-aea1-b5f4afeead8d",
        "1.0.0",
        cases,
        thresholds={
            "failure_rate": 0.0,
            "runtime_increase": 0.2,
            "memory_increase": 0.2,
        },
        metadata={"release": "0.1.0-alpha.1", "real_inference": False},
    )
