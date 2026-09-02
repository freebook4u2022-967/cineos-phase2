"""Fail-closed A/B decision gate for connected-shot identity refresh.

This module does not render video and cannot create production evidence. It only
compares two already production-attested connected GPU benchmark receipts: the
validated predecessor-frame baseline and the experimental CINEOS continuity +
fresh-reference compositor. The candidate is promotable only when identity gains
are measurable without material regression in temporal stability, motion quality,
or artifact integrity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class ContinuityIdentityABError(RuntimeError):
    """Raised when A/B evidence is incomplete, mismatched, or non-production."""


_REQUIRED_METRICS = (
    "identity_similarity",
    "temporal_consistency",
    "motion_quality",
    "artifact_integrity",
)


@dataclass(frozen=True, slots=True)
class ContinuityIdentityABDecision:
    """Auditable promotion decision for the experimental conditioning strategy."""

    promotable: bool
    baseline_chain_sha256: str
    candidate_chain_sha256: str
    shot_count: int
    baseline_means: dict[str, float]
    candidate_means: dict[str, float]
    deltas: dict[str, float]
    failed_criteria: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cineos-continuity-identity-ab-decision/0.1",
            "promotable": self.promotable,
            "baseline_chain_sha256": self.baseline_chain_sha256,
            "candidate_chain_sha256": self.candidate_chain_sha256,
            "shot_count": self.shot_count,
            "baseline_means": dict(self.baseline_means),
            "candidate_means": dict(self.candidate_means),
            "deltas": dict(self.deltas),
            "failed_criteria": list(self.failed_criteria),
        }


def _production_reports(receipt: Mapping[str, Any], *, label: str) -> tuple[Mapping[str, Any], ...]:
    if receipt.get("schema") != "cineos-gpu-connected-benchmark/0.2":
        raise ContinuityIdentityABError(f"{label} uses unsupported benchmark schema")
    if receipt.get("production_gpu_evidence") is not True:
        raise ContinuityIdentityABError(f"{label} is not production GPU evidence")
    if receipt.get("production_quality_evidence") is not True:
        raise ContinuityIdentityABError(f"{label} is not production measured QC evidence")
    if receipt.get("evidence_tier") != "production-gpu-quality-gated":
        raise ContinuityIdentityABError(f"{label} does not have the required evidence tier")
    reports = receipt.get("quality_reports")
    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes)):
        raise ContinuityIdentityABError(f"{label} quality_reports must be a sequence")
    if not 5 <= len(reports) <= 10:
        raise ContinuityIdentityABError(f"{label} must contain 5-10 measured shots")
    return tuple(reports)


def _metric_rows(reports: Sequence[Mapping[str, Any]], *, label: str) -> tuple[dict[str, float], ...]:
    rows: list[dict[str, float]] = []
    identities: set[tuple[Any, Any]] = set()
    for index, report in enumerate(reports):
        if not isinstance(report, Mapping) or report.get("accepted") is not True:
            raise ContinuityIdentityABError(f"{label} shot {index} is not accepted measured QC")
        identity = (report.get("scene_id"), report.get("shot_id"))
        if None in identity or identity in identities:
            raise ContinuityIdentityABError(f"{label} shot identities are missing or duplicated")
        identities.add(identity)
        measurement = report.get("measurement")
        if not isinstance(measurement, Mapping):
            raise ContinuityIdentityABError(f"{label} shot {index} has no measurement")
        metrics = measurement.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ContinuityIdentityABError(f"{label} shot {index} has no metric mapping")
        row: dict[str, float] = {}
        for name in _REQUIRED_METRICS:
            value = metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContinuityIdentityABError(f"{label} shot {index} missing numeric {name}")
            numeric = float(value)
            if not 0.0 <= numeric <= 1.0:
                raise ContinuityIdentityABError(f"{label} shot {index} {name} is outside [0, 1]")
            row[name] = numeric
        rows.append(row)
    return tuple(rows)


def evaluate_continuity_identity_ab(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    minimum_identity_gain: float = 0.02,
    maximum_identity_shot_regression: float = 0.03,
    maximum_temporal_regression: float = 0.01,
    maximum_motion_regression: float = 0.01,
    maximum_artifact_regression: float = 0.0,
) -> ContinuityIdentityABDecision:
    """Decide whether fresh identity refresh is safe to promote after real GPU A/B.

    The two receipts must be independent production-quality-gated runs of the same
    profile and same ordered shot identities. Thresholds are intentionally strict:
    identity must improve on average while continuity, motion, and artifact integrity
    may not materially regress. This gate is decision evidence, not model evidence.
    """

    thresholds = (
        minimum_identity_gain,
        maximum_identity_shot_regression,
        maximum_temporal_regression,
        maximum_motion_regression,
        maximum_artifact_regression,
    )
    if any(isinstance(value, bool) or value < 0 for value in thresholds):
        raise ValueError("A/B thresholds must be non-negative numbers")

    baseline_reports = _production_reports(baseline, label="baseline")
    candidate_reports = _production_reports(candidate, label="candidate")
    if baseline.get("profile_id") != candidate.get("profile_id"):
        raise ContinuityIdentityABError("A/B benchmark profiles differ")
    if len(baseline_reports) != len(candidate_reports):
        raise ContinuityIdentityABError("A/B shot counts differ")
    baseline_chain = baseline.get("chain_sha256")
    candidate_chain = candidate.get("chain_sha256")
    if not isinstance(baseline_chain, str) or len(baseline_chain) != 64:
        raise ContinuityIdentityABError("baseline chain digest is invalid")
    if not isinstance(candidate_chain, str) or len(candidate_chain) != 64:
        raise ContinuityIdentityABError("candidate chain digest is invalid")
    if baseline_chain == candidate_chain:
        raise ContinuityIdentityABError("A/B runs reused the same rendered chain")

    baseline_ids = [(item.get("scene_id"), item.get("shot_id")) for item in baseline_reports]
    candidate_ids = [(item.get("scene_id"), item.get("shot_id")) for item in candidate_reports]
    if baseline_ids != candidate_ids:
        raise ContinuityIdentityABError("A/B runs do not cover the same ordered shots")

    baseline_rows = _metric_rows(baseline_reports, label="baseline")
    candidate_rows = _metric_rows(candidate_reports, label="candidate")
    means = lambda rows, name: sum(row[name] for row in rows) / len(rows)
    baseline_means = {name: means(baseline_rows, name) for name in _REQUIRED_METRICS}
    candidate_means = {name: means(candidate_rows, name) for name in _REQUIRED_METRICS}
    deltas = {name: candidate_means[name] - baseline_means[name] for name in _REQUIRED_METRICS}

    failed: list[str] = []
    if deltas["identity_similarity"] < minimum_identity_gain:
        failed.append("insufficient_mean_identity_gain")
    if any(
        candidate["identity_similarity"] < baseline["identity_similarity"] - maximum_identity_shot_regression
        for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True)
    ):
        failed.append("per_shot_identity_regression")
    if deltas["temporal_consistency"] < -maximum_temporal_regression:
        failed.append("temporal_regression")
    if deltas["motion_quality"] < -maximum_motion_regression:
        failed.append("motion_regression")
    if deltas["artifact_integrity"] < -maximum_artifact_regression:
        failed.append("artifact_integrity_regression")

    return ContinuityIdentityABDecision(
        promotable=not failed,
        baseline_chain_sha256=baseline_chain,
        candidate_chain_sha256=candidate_chain,
        shot_count=len(baseline_rows),
        baseline_means=baseline_means,
        candidate_means=candidate_means,
        deltas=deltas,
        failed_criteria=tuple(failed),
    )


__all__ = [
    "ContinuityIdentityABDecision",
    "ContinuityIdentityABError",
    "evaluate_continuity_identity_ab",
]
