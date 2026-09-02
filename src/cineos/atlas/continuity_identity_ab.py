"""Fail-closed A/B decision gate for connected-shot identity refresh.

This module does not render video and cannot create production evidence. It only
compares two already production-attested connected GPU benchmark receipts. The
baseline must carry the predecessor-terminal-frame strategy and the candidate must
carry the exact first-party experimental CINEOS fresh-reference compositor in GPU
runtime provenance. The candidate is promotable only when identity gains are
measurable without material regression in temporal stability, motion quality, or
artifact integrity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .production_continuity_identity import (
    CONTINUITY_IDENTITY_ADAPTER_ID,
    CONTINUITY_IDENTITY_ADAPTER_VERSION,
)
from .production_continuity_identity_runtime import (
    CONTINUITY_IDENTITY_RUNTIME_SCHEMA,
)


class ContinuityIdentityABError(RuntimeError):
    """Raised when A/B evidence is incomplete, mismatched, or non-production."""


_REQUIRED_METRICS = (
    "identity_similarity",
    "temporal_consistency",
    "motion_quality",
    "artifact_integrity",
)
_BASELINE_MODE = "predecessor_terminal_frame_baseline"
_CANDIDATE_MODE = "predecessor_terminal_frame_plus_fresh_references"


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
            "schema": "cineos-continuity-identity-ab-decision/0.2",
            "promotable": self.promotable,
            "baseline_chain_sha256": self.baseline_chain_sha256,
            "candidate_chain_sha256": self.candidate_chain_sha256,
            "shot_count": self.shot_count,
            "baseline_means": dict(self.baseline_means),
            "candidate_means": dict(self.candidate_means),
            "deltas": dict(self.deltas),
            "failed_criteria": list(self.failed_criteria),
        }


def _production_reports(
    receipt: Mapping[str, Any], *, label: str
) -> tuple[Mapping[str, Any], ...]:
    if receipt.get("schema") != "cineos-gpu-connected-benchmark/0.2":
        raise ContinuityIdentityABError(f"{label} uses unsupported benchmark schema")
    if receipt.get("production_gpu_evidence") is not True:
        raise ContinuityIdentityABError(f"{label} is not production GPU evidence")
    if receipt.get("production_quality_evidence") is not True:
        raise ContinuityIdentityABError(
            f"{label} is not production measured QC evidence"
        )
    if receipt.get("evidence_tier") != "production-gpu-quality-gated":
        raise ContinuityIdentityABError(
            f"{label} does not have the required evidence tier"
        )
    reports = receipt.get("quality_reports")
    if not isinstance(reports, Sequence) or isinstance(reports, (str, bytes)):
        raise ContinuityIdentityABError(f"{label} quality_reports must be a sequence")
    if not 5 <= len(reports) <= 10:
        raise ContinuityIdentityABError(f"{label} must contain 5-10 measured shots")
    return tuple(reports)


def _strategy_for_shot(
    shot: Mapping[str, Any], *, label: str, index: int
) -> Mapping[str, Any]:
    runtime = shot.get("runtime_provenance")
    if not isinstance(runtime, Mapping):
        raise ContinuityIdentityABError(
            f"{label} shot {index} has no GPU runtime provenance"
        )
    if runtime.get("production_default_runtime") is not True:
        raise ContinuityIdentityABError(
            f"{label} shot {index} is not default production execution"
        )
    strategy = runtime.get("continuity_identity_strategy")
    if not isinstance(strategy, Mapping):
        raise ContinuityIdentityABError(
            f"{label} shot {index} has no continuity identity strategy provenance"
        )
    if strategy.get("schema") != CONTINUITY_IDENTITY_RUNTIME_SCHEMA:
        raise ContinuityIdentityABError(
            f"{label} shot {index} has unsupported continuity strategy schema"
        )
    return strategy


def _validate_strategy(
    receipt: Mapping[str, Any], *, label: str, expected_mode: str
) -> None:
    shots = receipt.get("shots")
    if not isinstance(shots, Sequence) or isinstance(shots, (str, bytes)):
        raise ContinuityIdentityABError(f"{label} has no serialized shot evidence")
    if not 5 <= len(shots) <= 10:
        raise ContinuityIdentityABError(f"{label} must contain 5-10 GPU shot receipts")

    for index, shot in enumerate(shots):
        if not isinstance(shot, Mapping):
            raise ContinuityIdentityABError(
                f"{label} shot {index} GPU evidence is not an object"
            )
        strategy = _strategy_for_shot(shot, label=label, index=index)
        if strategy.get("mode") != expected_mode:
            raise ContinuityIdentityABError(
                f"{label} shot {index} uses unexpected continuity identity strategy"
            )
        if expected_mode == _BASELINE_MODE:
            if strategy.get("adapter_id") is not None:
                raise ContinuityIdentityABError(
                    f"{label} baseline shot {index} unexpectedly declares an adapter"
                )
            if strategy.get("experimental") is not False:
                raise ContinuityIdentityABError(
                    f"{label} baseline shot {index} is marked experimental"
                )
        else:
            if strategy.get("adapter_id") != CONTINUITY_IDENTITY_ADAPTER_ID:
                raise ContinuityIdentityABError(
                    f"{label} candidate shot {index} uses an unrecognized adapter"
                )
            if strategy.get("adapter_version") != CONTINUITY_IDENTITY_ADAPTER_VERSION:
                raise ContinuityIdentityABError(
                    f"{label} candidate shot {index} adapter version differs"
                )
            if strategy.get("experimental") is not True:
                raise ContinuityIdentityABError(
                    f"{label} candidate shot {index} must remain experimental"
                )


def _metric_rows(
    reports: Sequence[Mapping[str, Any]], *, label: str
) -> tuple[dict[str, float], ...]:
    rows: list[dict[str, float]] = []
    identities: set[tuple[Any, Any]] = set()
    for index, report in enumerate(reports):
        if not isinstance(report, Mapping) or report.get("accepted") is not True:
            raise ContinuityIdentityABError(
                f"{label} shot {index} is not accepted measured QC"
            )
        identity = (report.get("scene_id"), report.get("shot_id"))
        if None in identity or identity in identities:
            raise ContinuityIdentityABError(
                f"{label} shot identities are missing or duplicated"
            )
        identities.add(identity)
        measurement = report.get("measurement")
        if not isinstance(measurement, Mapping):
            raise ContinuityIdentityABError(f"{label} shot {index} has no measurement")
        metrics = measurement.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ContinuityIdentityABError(
                f"{label} shot {index} has no metric mapping"
            )
        row: dict[str, float] = {}
        for name in _REQUIRED_METRICS:
            value = metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ContinuityIdentityABError(
                    f"{label} shot {index} missing numeric {name}"
                )
            numeric = float(value)
            if not 0.0 <= numeric <= 1.0:
                raise ContinuityIdentityABError(
                    f"{label} shot {index} {name} is outside [0, 1]"
                )
            row[name] = numeric
        rows.append(row)
    return tuple(rows)


def _mean(rows: Sequence[Mapping[str, float]], name: str) -> float:
    return sum(row[name] for row in rows) / len(rows)


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


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
    profile and ordered shot identities. Their serialized GPU runtime provenance
    must prove that the baseline used predecessor-only continuity and the candidate
    used the exact current first-party CINEOS experimental compositor. Thresholds
    are intentionally strict: identity must improve while continuity, motion, and
    artifact integrity may not materially regress. This is decision evidence, not
    a claim that the external foundation weights are CINEOS-native.
    """

    thresholds = (
        minimum_identity_gain,
        maximum_identity_shot_regression,
        maximum_temporal_regression,
        maximum_motion_regression,
        maximum_artifact_regression,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
        for value in thresholds
    ):
        raise ValueError("A/B thresholds must be non-negative numbers")

    baseline_reports = _production_reports(baseline, label="baseline")
    candidate_reports = _production_reports(candidate, label="candidate")
    _validate_strategy(baseline, label="baseline", expected_mode=_BASELINE_MODE)
    _validate_strategy(candidate, label="candidate", expected_mode=_CANDIDATE_MODE)

    if baseline.get("profile_id") != candidate.get("profile_id"):
        raise ContinuityIdentityABError("A/B benchmark profiles differ")
    if len(baseline_reports) != len(candidate_reports):
        raise ContinuityIdentityABError("A/B shot counts differ")
    baseline_chain = baseline.get("chain_sha256")
    candidate_chain = candidate.get("chain_sha256")
    if not _valid_sha256(baseline_chain):
        raise ContinuityIdentityABError("baseline chain digest is invalid")
    if not _valid_sha256(candidate_chain):
        raise ContinuityIdentityABError("candidate chain digest is invalid")
    if baseline_chain == candidate_chain:
        raise ContinuityIdentityABError("A/B runs reused the same rendered chain")

    baseline_ids = [
        (item.get("scene_id"), item.get("shot_id")) for item in baseline_reports
    ]
    candidate_ids = [
        (item.get("scene_id"), item.get("shot_id")) for item in candidate_reports
    ]
    if baseline_ids != candidate_ids:
        raise ContinuityIdentityABError("A/B runs do not cover the same ordered shots")

    baseline_rows = _metric_rows(baseline_reports, label="baseline")
    candidate_rows = _metric_rows(candidate_reports, label="candidate")
    baseline_means = {name: _mean(baseline_rows, name) for name in _REQUIRED_METRICS}
    candidate_means = {name: _mean(candidate_rows, name) for name in _REQUIRED_METRICS}
    deltas = {
        name: candidate_means[name] - baseline_means[name] for name in _REQUIRED_METRICS
    }

    failed: list[str] = []
    if deltas["identity_similarity"] < minimum_identity_gain:
        failed.append("insufficient_mean_identity_gain")
    if any(
        candidate_row["identity_similarity"]
        < baseline_row["identity_similarity"] - maximum_identity_shot_regression
        for baseline_row, candidate_row in zip(
            baseline_rows, candidate_rows, strict=True
        )
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
