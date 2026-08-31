"""Auditable CINEOS quality gate for connected foundation-video sequences.

This module does not claim that an external foundation model is CINEOS-native.
It owns the benchmark decision around generated artifacts so that identity,
temporal, artifact, and motion evidence are evaluated consistently before the
existing reject/rerender loop accepts a shot into a film sequence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CORE_METRICS = (
    "identity_similarity",
    "temporal_consistency",
    "artifact_integrity",
    "motion_quality",
)
PRODUCTION_MEASUREMENT_SCHEMA = "cineos-sequence-quality-measurement/0.1"


@dataclass(frozen=True, slots=True)
class SequenceQualityPolicy:
    """Versioned thresholds for production-style connected-shot acceptance."""

    identity_floor: float = 0.78
    temporal_floor: float = 0.76
    artifact_floor: float = 0.90
    motion_floor: float = 0.72
    overall_floor: float = 0.80

    def __post_init__(self) -> None:
        values = {
            "identity_floor": self.identity_floor,
            "temporal_floor": self.temporal_floor,
            "artifact_floor": self.artifact_floor,
            "motion_floor": self.motion_floor,
            "overall_floor": self.overall_floor,
        }
        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    def snapshot(self) -> dict[str, float | str]:
        return {
            "schema": "cineos-sequence-quality-policy/0.1",
            "identity_floor": self.identity_floor,
            "temporal_floor": self.temporal_floor,
            "artifact_floor": self.artifact_floor,
            "motion_floor": self.motion_floor,
            "overall_floor": self.overall_floor,
        }


class SequenceQualityError(RuntimeError):
    """Raised when benchmark evidence is incomplete or malformed."""


def _validated_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    missing = [name for name in CORE_METRICS if name not in report]
    if missing:
        raise SequenceQualityError(
            "quality metric extractor missing required metric(s): " + ", ".join(missing)
        )

    metrics: dict[str, float] = {}
    for name, value in report.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise SequenceQualityError(
                f"quality metric {name!r} must be between 0 and 1"
            )
        metrics[name] = numeric
    return metrics


def _overall_score(metrics: Mapping[str, float]) -> float:
    """Weight identity/temporal evidence above secondary aesthetic metrics."""

    core = (
        0.32 * metrics["identity_similarity"]
        + 0.30 * metrics["temporal_consistency"]
        + 0.20 * metrics["artifact_integrity"]
        + 0.18 * metrics["motion_quality"]
    )
    optional_names = (
        "anatomy_quality",
        "object_interaction_quality",
        "dialogue_lip_sync",
    )
    optional = [metrics[name] for name in optional_names if name in metrics]
    if not optional:
        return core
    optional_mean = sum(optional) / len(optional)
    return 0.85 * core + 0.15 * optional_mean


def _quality_report(
    metrics: Mapping[str, float], policy: SequenceQualityPolicy
) -> dict[str, Any]:
    overall = _overall_score(metrics)
    failures: list[str] = []
    directives: list[str] = []
    thresholds = {
        "identity_similarity": policy.identity_floor,
        "temporal_consistency": policy.temporal_floor,
        "artifact_integrity": policy.artifact_floor,
        "motion_quality": policy.motion_floor,
    }
    directive_by_metric = {
        "identity_similarity": "preserve approved character identity and facial structure",
        "temporal_consistency": "reduce cross-frame and cross-shot temporal drift",
        "artifact_integrity": "remove corruption, malformed frames, and export artifacts",
        "motion_quality": "stabilize physically plausible subject and camera motion",
    }
    for name, threshold in thresholds.items():
        if metrics[name] < threshold:
            failures.append(name)
            directives.append(directive_by_metric[name])

    if overall < policy.overall_floor:
        failures.append("overall_score")
        directives.append(
            "increase total shot quality without sacrificing identity continuity"
        )

    accepted = not failures
    return {
        "schema": "cineos-sequence-quality-report/0.1",
        "accepted": accepted,
        "decision": "accept" if accepted else "reject",
        "score": overall,
        "metrics": dict(metrics),
        "failed_metrics": failures,
        "directives": directives,
        "policy": policy.snapshot(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CineosSequenceQualityEvaluator:
    """Callable bridge from measured shot metrics to the rerender loop.

    ``metric_extractor`` must inspect the rendered artifact and return normalized
    metrics in [0, 1]. Core metrics are mandatory so a missing observer cannot
    silently turn into an acceptance. This generic evaluator is intentionally
    suitable for deterministic tests as well as research experiments; production
    milestone evidence should use :class:`ArtifactMeasuredSequenceQualityEvaluator`.
    """

    def __init__(
        self, metric_extractor: Any, policy: SequenceQualityPolicy | None = None
    ) -> None:
        if not callable(metric_extractor):
            raise TypeError("metric_extractor must be callable")
        self.metric_extractor = metric_extractor
        self.policy = policy or SequenceQualityPolicy()

    def __call__(
        self,
        output_path: str,
        *,
        shot: Any,
        attempt_index: int,
    ) -> dict[str, Any]:
        raw = self.metric_extractor(
            output_path,
            shot=shot,
            attempt_index=attempt_index,
        )
        if not isinstance(raw, Mapping):
            raise SequenceQualityError("quality metric extractor must return a mapping")
        metrics = _validated_metrics(raw)
        return _quality_report(metrics, self.policy)


class ArtifactMeasuredSequenceQualityEvaluator:
    """Production evaluator requiring metrics cryptographically bound to the video.

    A production metric extractor must inspect the actual rendered artifact and
    return a structured measurement envelope containing its observer identity,
    the SHA-256 of the exact artifact it measured, and normalized metrics. This
    prevents a synthetic lambda or stale report from being accepted as evidence
    for the real-GPU milestone while keeping the observer implementation pluggable
    for stronger identity, temporal, anatomy, interaction, and lip-sync models.
    """

    production_measurement_evidence = True

    def __init__(
        self, metric_extractor: Any, policy: SequenceQualityPolicy | None = None
    ) -> None:
        if not callable(metric_extractor):
            raise TypeError("metric_extractor must be callable")
        self.metric_extractor = metric_extractor
        self.policy = policy or SequenceQualityPolicy()

    def __call__(
        self,
        output_path: str,
        *,
        shot: Any,
        attempt_index: int,
    ) -> dict[str, Any]:
        artifact = Path(output_path)
        if not artifact.is_file():
            raise SequenceQualityError(
                f"production quality artifact does not exist: {artifact}"
            )
        artifact_sha256 = _sha256_file(artifact)
        raw = self.metric_extractor(
            output_path,
            shot=shot,
            attempt_index=attempt_index,
        )
        if not isinstance(raw, Mapping):
            raise SequenceQualityError("quality metric extractor must return a mapping")
        if raw.get("schema") != PRODUCTION_MEASUREMENT_SCHEMA:
            raise SequenceQualityError(
                "unsupported production quality measurement schema"
            )
        observer_id = raw.get("observer_id")
        if not isinstance(observer_id, str) or not observer_id.strip():
            raise SequenceQualityError(
                "production quality measurement requires a non-empty observer_id"
            )
        measured_sha256 = raw.get("artifact_sha256")
        if measured_sha256 != artifact_sha256:
            raise SequenceQualityError(
                "production quality measurement artifact SHA-256 does not match rendered output"
            )
        raw_metrics = raw.get("metrics")
        if not isinstance(raw_metrics, Mapping):
            raise SequenceQualityError(
                "production quality measurement requires a metrics mapping"
            )
        metrics = _validated_metrics(raw_metrics)
        report = _quality_report(metrics, self.policy)
        report["production_measurement_evidence"] = True
        report["measurement"] = {
            "schema": PRODUCTION_MEASUREMENT_SCHEMA,
            "observer_id": observer_id.strip(),
            "artifact_sha256": artifact_sha256,
        }
        return report


__all__ = [
    "CORE_METRICS",
    "PRODUCTION_MEASUREMENT_SCHEMA",
    "ArtifactMeasuredSequenceQualityEvaluator",
    "CineosSequenceQualityEvaluator",
    "SequenceQualityError",
    "SequenceQualityPolicy",
]
