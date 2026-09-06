"""Unified fail-closed attestation for connected production render evidence.

This module does not claim that an external pretrained foundation is CINEOS-native.
It verifies that CINEOS orchestration has produced a connected sequence whose GPU
runtime provenance, measured visual QC, artifact-bound continuity lineage, and
measured cross-shot transition QC all refer to the same accepted render receipts.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .connected_continuity_evidence import (
    ConnectedContinuityEvidenceError,
    validate_connected_visual_continuity,
)
from .gpu_connected_benchmark import GPUConnectedBenchmarkReceipt
from .transition_quality import TRANSITION_QUALITY_SCHEMA


class ProductionConnectedEvidenceError(RuntimeError):
    """Raised when a benchmark cannot support a production connected-film claim."""


@dataclass(frozen=True, slots=True)
class ProductionConnectedEvidence:
    """Auditable result of the unified connected-production evidence gate."""

    benchmark_id: str
    profile_id: str
    origin: str
    shot_count: int
    chain_sha256: str
    runtime_valid: bool
    quality_valid: bool
    continuity_valid: bool
    transition_quality_valid: bool
    continuity_provenance: tuple[dict[str, Any], ...]

    @property
    def accepted(self) -> bool:
        return (
            self.runtime_valid
            and self.quality_valid
            and self.continuity_valid
            and self.transition_quality_valid
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cineos-production-connected-evidence/0.3",
            "benchmark_id": self.benchmark_id,
            "profile_id": self.profile_id,
            "origin": self.origin,
            "shot_count": self.shot_count,
            "chain_sha256": self.chain_sha256,
            "runtime_valid": self.runtime_valid,
            "quality_valid": self.quality_valid,
            "continuity_valid": self.continuity_valid,
            "transition_quality_valid": self.transition_quality_valid,
            "accepted": self.accepted,
            "continuity_provenance": list(self.continuity_provenance),
        }


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductionConnectedEvidenceError(
            f"production transition evidence requires {field}"
        )
    return value.strip()


def _required_unit_metric(value: Any, *, field: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionConnectedEvidenceError(
            f"production transition evidence {index} metric {field} must be numeric"
        )
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ProductionConnectedEvidenceError(
            f"production transition evidence {index} metric {field} must be finite "
            "and between 0 and 1"
        )
    return normalized


def _validate_transition_quality_manifest(
    benchmark: GPUConnectedBenchmarkReceipt,
) -> None:
    """Require measured artifact-bound QC for every connected shot boundary."""

    manifest = Path(benchmark.manifest_path)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionConnectedEvidenceError(
            "production connected evidence cannot read its benchmark manifest"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ProductionConnectedEvidenceError(
            "production connected benchmark manifest must be a JSON object"
        )
    if payload.get("chain_sha256") != benchmark.chain_sha256:
        raise ProductionConnectedEvidenceError(
            "production connected benchmark manifest chain hash does not match receipt"
        )

    gate = payload.get("quality_retry_gate")
    if not isinstance(gate, Mapping):
        raise ProductionConnectedEvidenceError(
            "production connected evidence requires a transition quality gate"
        )
    if gate.get("transition_gate_applied") is not True:
        raise ProductionConnectedEvidenceError(
            "production connected evidence requires measured transition QC"
        )

    expected = len(benchmark.shot_receipts) - 1
    if gate.get("accepted_transition_count") != expected:
        raise ProductionConnectedEvidenceError(
            "production transition evidence does not cover every shot boundary"
        )
    transitions = gate.get("accepted_transitions")
    if not isinstance(transitions, list) or len(transitions) != expected:
        raise ProductionConnectedEvidenceError(
            "production transition evidence list does not cover every shot boundary"
        )

    for index, report in enumerate(transitions):
        if not isinstance(report, Mapping):
            raise ProductionConnectedEvidenceError(
                f"production transition evidence {index} must be a mapping"
            )
        if report.get("schema") != TRANSITION_QUALITY_SCHEMA:
            raise ProductionConnectedEvidenceError(
                f"production transition evidence {index} has unsupported schema"
            )
        if report.get("production_measurement_evidence") is not True:
            raise ProductionConnectedEvidenceError(
                f"production transition evidence {index} is not measured evidence"
            )
        if report.get("accepted") is not True:
            raise ProductionConnectedEvidenceError(
                f"production transition evidence {index} was not accepted"
            )
        _required_text(report.get("observer_id"), field="observer_id")
        sample_count = report.get("measured_sample_count")
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count <= 0
        ):
            raise ProductionConnectedEvidenceError(
                f"production transition evidence {index} has no measured samples"
            )

        metrics = report.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ProductionConnectedEvidenceError(
                f"production transition evidence {index} requires measured metrics"
            )
        _required_unit_metric(
            metrics.get("visual_seam_similarity"),
            field="visual_seam_similarity",
            index=index,
        )
        _required_unit_metric(
            metrics.get("motion_boundary_consistency"),
            field="motion_boundary_consistency",
            index=index,
        )
        failed_metrics = report.get("failed_metrics")
        if not isinstance(failed_metrics, list) or failed_metrics:
            raise ProductionConnectedEvidenceError(
                f"production transition evidence {index} accepted report contains "
                "failed metrics"
            )

        previous_receipt = benchmark.shot_receipts[index]
        current_receipt = benchmark.shot_receipts[index + 1]
        previous_sha = _required_text(
            getattr(previous_receipt, "output_sha256", None),
            field="previous artifact SHA-256",
        )
        current_sha = _required_text(
            getattr(current_receipt, "output_sha256", None),
            field="current artifact SHA-256",
        )
        if report.get("previous_output_sha256") != previous_sha:
            raise ProductionConnectedEvidenceError(
                f"production transition evidence {index} predecessor hash mismatch"
            )
        if report.get("current_output_sha256") != current_sha:
            raise ProductionConnectedEvidenceError(
                f"production transition evidence {index} current artifact hash mismatch"
            )

        previous_result = getattr(previous_receipt, "result", None)
        current_result = getattr(current_receipt, "result", None)
        if previous_result is None or current_result is None:
            raise ProductionConnectedEvidenceError(
                "production transition evidence requires render result lineage"
            )
        expected_identity = {
            "previous_scene_id": getattr(previous_result, "scene_id", None),
            "previous_shot_id": getattr(previous_result, "shot_id", None),
            "current_scene_id": getattr(current_result, "scene_id", None),
            "current_shot_id": getattr(current_result, "shot_id", None),
        }
        for field, expected_value in expected_identity.items():
            if report.get(field) != expected_value:
                raise ProductionConnectedEvidenceError(
                    f"production transition evidence {index} {field} mismatch"
                )


def validate_production_connected_evidence(
    benchmark: GPUConnectedBenchmarkReceipt,
) -> ProductionConnectedEvidence:
    """Validate one benchmark as genuine connected production evidence.

    The gate intentionally requires four independent attestations: default CUDA
    runtime provenance, artifact-bound measured QC for every accepted shot,
    cryptographically bound predecessor terminal-frame lineage, and measured
    artifact-bound transition QC for every adjacent shot boundary.
    """

    if not isinstance(benchmark, GPUConnectedBenchmarkReceipt):
        raise TypeError("benchmark must be a GPUConnectedBenchmarkReceipt")
    shot_count = len(benchmark.shot_receipts)
    if not 5 <= shot_count <= 10:
        raise ProductionConnectedEvidenceError(
            "production connected evidence requires between 5 and 10 render receipts"
        )
    if not benchmark.production_gpu_evidence:
        raise ProductionConnectedEvidenceError(
            "production connected evidence requires default CUDA runtime provenance"
        )
    if not benchmark.production_quality_evidence:
        raise ProductionConnectedEvidenceError(
            "production connected evidence requires measured artifact-bound QC "
            "for every shot"
        )

    try:
        continuity = validate_connected_visual_continuity(benchmark.shot_receipts)
    except ConnectedContinuityEvidenceError as exc:
        raise ProductionConnectedEvidenceError(
            f"production connected evidence failed visual continuity validation: {exc}"
        ) from exc

    _validate_transition_quality_manifest(benchmark)

    evidence = ProductionConnectedEvidence(
        benchmark_id=benchmark.benchmark_id,
        profile_id=benchmark.profile_id,
        origin=benchmark.origin,
        shot_count=shot_count,
        chain_sha256=benchmark.chain_sha256,
        runtime_valid=True,
        quality_valid=True,
        continuity_valid=True,
        transition_quality_valid=True,
        continuity_provenance=continuity,
    )
    if not evidence.accepted:
        raise ProductionConnectedEvidenceError(
            "production connected evidence did not satisfy the unified gate"
        )
    return evidence


def production_connected_evidence(
    benchmark: GPUConnectedBenchmarkReceipt,
) -> bool:
    """Return whether a connected benchmark satisfies the unified production gate."""

    try:
        validate_production_connected_evidence(benchmark)
    except (ProductionConnectedEvidenceError, TypeError):
        return False
    return True


__all__ = [
    "ProductionConnectedEvidence",
    "ProductionConnectedEvidenceError",
    "production_connected_evidence",
    "validate_production_connected_evidence",
]
