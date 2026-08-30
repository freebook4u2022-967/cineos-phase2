"""Fail-closed validation for production real-inference benchmark evidence.

This module does not execute a model. It validates the evidence emitted by the
production GPU path so a competitive benchmark cannot be declared passing from
synthetic metrics, missing media, or ambiguous foundation provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .case import BenchmarkCase
from .exceptions import BenchmarkError
from .metrics import MetricStatus
from .report import CaseResult


def validate_real_inference_evidence(
    case: BenchmarkCase,
    result: CaseResult,
    output_dir: str | Path,
    *,
    foundation: Mapping[str, object],
) -> None:
    """Validate one production GPU case before it can count toward release.

    Competitive cases require real inference, non-empty declared artifacts,
    threshold metrics produced by measurement (not estimates/manual review), and
    explicit external-pretrained-foundation provenance. The function raises
    ``BenchmarkError`` on the first invalid condition and otherwise returns None.
    """

    if case.hardware_requirements.get("real_inference") is not True:
        raise BenchmarkError("case is not declared as a real-inference benchmark")
    if result.case_id != case.case_id:
        raise BenchmarkError("benchmark result case_id does not match case contract")
    if not result.passed:
        raise BenchmarkError("real-inference benchmark result did not pass")

    root = Path(output_dir).resolve()
    declared_outputs = set(result.outputs)
    for expected in case.expected_outputs:
        if expected not in declared_outputs:
            raise BenchmarkError(f"missing declared benchmark output: {expected}")
        artifact = (root / expected).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as exc:
            raise BenchmarkError("benchmark output escapes the case output directory") from exc
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise BenchmarkError(f"missing or empty benchmark artifact: {expected}")

    metric_by_name = {metric.name: metric for metric in result.metrics}
    for name, threshold in case.validation_thresholds.items():
        metric = metric_by_name.get(name)
        if metric is None:
            raise BenchmarkError(f"missing required measured metric: {name}")
        if metric.status is not MetricStatus.MEASURED:
            raise BenchmarkError(f"required metric is not measured: {name}")
        if isinstance(metric.value, bool) or not isinstance(metric.value, (int, float)):
            raise BenchmarkError(f"required metric is not numeric: {name}")
        if float(metric.value) < threshold:
            raise BenchmarkError(
                f"metric {name}={float(metric.value):.4f} is below threshold {threshold:.4f}"
            )

    origin = foundation.get("origin")
    model_id = foundation.get("model_id")
    if origin != "external_pretrained_foundation":
        raise BenchmarkError(
            "competitive real inference must declare external_pretrained_foundation origin"
        )
    if not isinstance(model_id, str) or not model_id.strip():
        raise BenchmarkError("competitive real inference must declare foundation model_id")
