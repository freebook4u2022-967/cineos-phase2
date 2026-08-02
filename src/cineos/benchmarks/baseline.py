"""Immutable, manually approved benchmark baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .exceptions import BenchmarkError
from .report import BenchmarkReport


@dataclass(frozen=True, slots=True)
class Baseline:
    baseline_id: str
    report: dict[str, object]
    approved: bool = False
    approved_by: str | None = None


def create_baseline(report: BenchmarkReport, path: Path) -> Baseline:
    if path.exists():
        raise BenchmarkError("refusing to overwrite historical baseline")
    baseline = Baseline(report.suite_hash[:16], asdict(report))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(baseline), sort_keys=True, indent=2) + "\n")
    return baseline


def load_baseline(path: Path) -> Baseline:
    return Baseline(**json.loads(path.read_text(encoding="utf-8")))


def approve_baseline(path: Path, approver: str) -> Baseline:
    baseline = load_baseline(path)
    approved = Baseline(baseline.baseline_id, baseline.report, True, approver)
    path.write_text(json.dumps(asdict(approved), sort_keys=True, indent=2) + "\n")
    return approved
