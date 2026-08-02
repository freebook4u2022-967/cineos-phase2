"""Deterministic benchmark runner; it performs no real AI inference."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .metrics import METRIC_NAMES, Metric, MetricStatus
from .report import BenchmarkReport, CaseResult
from .serializer import save
from .suite import BenchmarkSuite


class BenchmarkRunner:
    def run(
        self,
        suite: BenchmarkSuite,
        output_dir: Path,
        *,
        mandatory_only: bool = False,
        include_slow: bool = False,
        hardware_profile: str = "cpu",
        renderer: str | None = None,
        dry_run: bool = False,
    ) -> BenchmarkReport:
        results = []
        for case in suite.cases:
            if (mandatory_only and not case.mandatory) or (
                case.slow and not include_slow
            ):
                continue
            fixture = Path(case.project_fixture)
            fixture_ok = fixture.is_file()
            digest = hashlib.sha256(
                f"{case.case_id}:{case.deterministic_seed}".encode()
            ).hexdigest()
            metrics = tuple(
                (
                    Metric(name, fixture_ok, MetricStatus.MEASURED)
                    if name == "execution_success"
                    else Metric(name, status=MetricStatus.UNSUPPORTED)
                )
                for name in METRIC_NAMES
            )
            results.append(
                CaseResult(
                    case.case_id,
                    fixture_ok or dry_run,
                    metrics,
                    deterministic_hash=digest,
                )
            )
        report = BenchmarkReport(
            suite.suite_id,
            suite.suite_version,
            suite.content_hash,
            renderer or suite.renderer_profile,
            hardware_profile,
            tuple(results),
            {"dry_run": dry_run},
        )
        save(report, output_dir / "report.json")
        return report
