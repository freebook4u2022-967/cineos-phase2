"""Benchmark regression classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .baseline import Baseline
from .report import BenchmarkReport


class Severity(StrEnum):
    INFORMATIONAL = "informational"
    WARNING = "warning"
    BLOCKING = "blocking"


@dataclass(frozen=True, slots=True)
class Regression:
    kind: str
    severity: Severity
    case_id: str | None
    message: str


def compare(report: BenchmarkReport, baseline: Baseline) -> tuple[Regression, ...]:
    old = {item["case_id"]: item for item in baseline.report.get("results", [])}
    findings = []
    for result in report.results:
        previous = old.get(result.case_id)
        if previous and previous.get("passed") and not result.passed:
            findings.append(
                Regression(
                    "increased_failure_rate",
                    Severity.BLOCKING,
                    result.case_id,
                    "previously passing case failed",
                )
            )
        if previous and previous.get("deterministic_hash") != result.deterministic_hash:
            findings.append(
                Regression(
                    "changed_deterministic_hash",
                    Severity.BLOCKING,
                    result.case_id,
                    "deterministic hash changed",
                )
            )
        if previous:
            old_metrics = {
                item["name"]: item.get("value") for item in previous.get("metrics", ())
            }
            new_metrics = {item.name: item.value for item in result.metrics}
            scores = (
                "identity_score",
                "wardrobe_continuity_score",
                "prop_continuity_score",
                "environment_continuity_score",
                "temporal_stability",
                "lip_sync_timing_accuracy",
                "audio_alignment",
            )
            for name in scores:
                old_value, new_value = old_metrics.get(name), new_metrics.get(name)
                if (
                    isinstance(old_value, (int, float))
                    and isinstance(new_value, (int, float))
                    and new_value < old_value
                ):
                    findings.append(
                        Regression(
                            "score_degradation",
                            Severity.WARNING,
                            result.case_id,
                            f"{name} decreased",
                        )
                    )
            resources = (
                ("runtime_per_shot", "longer_runtime"),
                ("total_build_time", "longer_runtime"),
                ("peak_ram", "higher_memory_use"),
                ("peak_vram", "higher_memory_use"),
            )
            for name, kind in resources:
                old_value, new_value = old_metrics.get(name), new_metrics.get(name)
                if (
                    isinstance(old_value, (int, float))
                    and isinstance(new_value, (int, float))
                    and new_value > old_value * 1.2
                ):
                    findings.append(
                        Regression(
                            kind,
                            Severity.WARNING,
                            result.case_id,
                            f"{name} increased by more than 20%",
                        )
                    )
            new_warnings = set(result.warnings) - set(previous.get("warnings", ()))
            findings.extend(
                Regression("new_warning", Severity.WARNING, result.case_id, item)
                for item in sorted(new_warnings)
            )
            if set(previous.get("outputs", ())) - set(result.outputs):
                findings.append(
                    Regression(
                        "missing_outputs",
                        Severity.BLOCKING,
                        result.case_id,
                        "baseline outputs are missing",
                    )
                )
    old_api = baseline.report.get("metadata", {}).get("public_api_hash")
    new_api = report.metadata.get("public_api_hash")
    if old_api and new_api and old_api != new_api:
        findings.append(
            Regression(
                "altered_public_api_behavior",
                Severity.BLOCKING,
                None,
                "public API behavior hash changed",
            )
        )
    return tuple(findings)


compare_against_baseline = compare
