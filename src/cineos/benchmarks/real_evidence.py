"""Fail-closed validation for production real-inference benchmark evidence.

This module does not execute a model. It validates the evidence emitted by the
production GPU path so a competitive benchmark cannot be declared passing from
synthetic metrics, missing media, malformed receipts, or ambiguous foundation
provenance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
    structurally valid benchmark evidence, threshold metrics produced by measurement
    (not estimates/manual review), and explicit external-pretrained-foundation
    provenance. The function raises ``BenchmarkError`` on the first invalid
    condition and otherwise returns None.
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
            raise BenchmarkError(
                "benchmark output escapes the case output directory"
            ) from exc
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise BenchmarkError(f"missing or empty benchmark artifact: {expected}")
        _validate_artifact_structure(expected, artifact)

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
        raise BenchmarkError(
            "competitive real inference must declare foundation model_id"
        )


def _validate_artifact_structure(name: str, artifact: Path) -> None:
    """Reject placeholder files before they can satisfy the production gate.

    JSON evidence must parse to an object. MP4 evidence must expose an ISO-BMFF
    ``ftyp`` box near the start of the file. This is intentionally lightweight and
    dependency-free: decode-level validation remains the responsibility of the
    production media probe, while this guard prevents arbitrary non-empty bytes from
    being accepted as real benchmark evidence.
    """

    suffix = artifact.suffix.lower()
    if suffix == ".json":
        payload = _read_json_object(artifact)
        if name == "report.json" and not payload:
            raise BenchmarkError("benchmark report JSON object is empty")
        return
    if suffix == ".mp4":
        _validate_mp4_container(artifact)


def _read_json_object(artifact: Path) -> dict[str, Any]:
    try:
        with artifact.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(
            f"benchmark JSON artifact is malformed: {artifact.name}"
        ) from exc
    if not isinstance(payload, dict):
        raise BenchmarkError(
            f"benchmark JSON artifact must contain an object: {artifact.name}"
        )
    return payload


def _validate_mp4_container(artifact: Path) -> None:
    try:
        with artifact.open("rb") as handle:
            header = handle.read(64)
    except OSError as exc:
        raise BenchmarkError(
            f"unable to read benchmark video artifact: {artifact.name}"
        ) from exc

    # ISO Base Media File Format starts with a small box whose type is commonly
    # ``ftyp`` at byte offset 4. Search the first 64 bytes to tolerate legal leading
    # boxes while still rejecting arbitrary placeholder text or raw frame dumps.
    if b"ftyp" not in header:
        raise BenchmarkError(
            f"benchmark video artifact is not an MP4/ISO-BMFF container: {artifact.name}"
        )
