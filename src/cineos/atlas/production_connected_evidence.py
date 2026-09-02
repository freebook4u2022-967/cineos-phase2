"""Unified fail-closed attestation for connected production render evidence.

This module does not claim that an external pretrained foundation is CINEOS-native.
It verifies that CINEOS orchestration has produced a connected sequence whose GPU
runtime provenance, measured visual QC, and artifact-bound continuity lineage all
refer to the same accepted render receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .connected_continuity_evidence import (
    ConnectedContinuityEvidenceError,
    validate_connected_visual_continuity,
)
from .gpu_connected_benchmark import GPUConnectedBenchmarkReceipt


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
    continuity_provenance: tuple[dict[str, Any], ...]

    @property
    def accepted(self) -> bool:
        return self.runtime_valid and self.quality_valid and self.continuity_valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cineos-production-connected-evidence/0.1",
            "benchmark_id": self.benchmark_id,
            "profile_id": self.profile_id,
            "origin": self.origin,
            "shot_count": self.shot_count,
            "chain_sha256": self.chain_sha256,
            "runtime_valid": self.runtime_valid,
            "quality_valid": self.quality_valid,
            "continuity_valid": self.continuity_valid,
            "accepted": self.accepted,
            "continuity_provenance": list(self.continuity_provenance),
        }


def validate_production_connected_evidence(
    benchmark: GPUConnectedBenchmarkReceipt,
) -> ProductionConnectedEvidence:
    """Validate one benchmark as genuine connected production evidence.

    The gate intentionally requires all three independent attestations:
    default CUDA runtime provenance, artifact-bound measured QC for every accepted
    shot, and cryptographically bound predecessor terminal-frame lineage.
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

    evidence = ProductionConnectedEvidence(
        benchmark_id=benchmark.benchmark_id,
        profile_id=benchmark.profile_id,
        origin=benchmark.origin,
        shot_count=shot_count,
        chain_sha256=benchmark.chain_sha256,
        runtime_valid=True,
        quality_valid=True,
        continuity_valid=True,
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
