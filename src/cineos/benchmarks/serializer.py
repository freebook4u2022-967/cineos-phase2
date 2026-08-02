"""Canonical JSON persistence for benchmark records."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .metrics import Metric, MetricStatus
from .report import BenchmarkReport, CaseResult


def dumps(value: object) -> str:
    return json.dumps(asdict(value), sort_keys=True, separators=(",", ":")) + "\n"


def save(value: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(value), encoding="utf-8")
    return path


def load_report(path: Path) -> BenchmarkReport:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["results"] = tuple(
        CaseResult(
            **{
                **item,
                "metrics": tuple(
                    Metric(**{**metric, "status": MetricStatus(metric["status"])})
                    for metric in item["metrics"]
                ),
                "warnings": tuple(item.get("warnings", ())),
                "outputs": tuple(item.get("outputs", ())),
            }
        )
        for item in raw["results"]
    )
    return BenchmarkReport(**raw)
