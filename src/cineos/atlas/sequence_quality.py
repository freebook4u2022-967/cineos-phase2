"""Auditable CINEOS quality gate for connected foundation-video sequences.

This module does not claim that an external foundation model is CINEOS-native.
It owns the benchmark decision around generated artifacts so that identity,
temporal, artifact, motion, and difficult-case evidence are evaluated consistently
before the existing reject/rerender loop accepts a shot into a film sequence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
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

# Competitive challenge declarations are promises about a specific visual behavior.
# They must therefore be backed by a dedicated measured metric. Generic identity,
# temporal, or motion scores are useful core evidence, but cannot prove locomotion,
# interaction, camera execution, lighting-transition fidelity, or physics by proxy.
CHALLENGE_METRIC_REQUIREMENTS = {
    "multi_character_interaction": "multi_character_interaction_quality",
    "hands_anatomy": "anatomy_quality",
    "walking_running": "locomotion_quality",
    "object_interaction": "object_interaction_quality",
    "fast_camera_movement": "camera_motion_quality",
    "lighting_changes": "lighting_transition_quality",
    "physics": "physics_plausibility",
    "dialogue_lip_sync": "dialogue_lip_sync",
    # The newer Seedance-style naming is accepted as an alias at the quality boundary.
    "dialogue": "dialogue_lip_sync",
}
_CHALLENGE_METADATA_KEYS = ("competitive_challenges", "benchmark_challenges")


@dataclass(frozen=True, slots=True)
class SequenceQualityPolicy:
    """Versioned thresholds for production-style connected-shot acceptance."""

    identity_floor: float = 0.78
    temporal_floor: float = 0.76
    artifact_floor: float = 0.90
    motion_floor: float = 0.72
    overall_floor: float = 0.80
    multi_character_interaction_floor: float = 0.76
    anatomy_floor: float = 0.78
    locomotion_floor: float = 0.74
    object_interaction_floor: float = 0.76
    camera_motion_floor: float = 0.72
    lighting_transition_floor: float = 0.74
    physics_plausibility_floor: float = 0.74
    dialogue_lip_sync_floor: float = 0.74

    def __post_init__(self) -> None:
        values = {
            "identity_floor": self.identity_floor,
            "temporal_floor": self.temporal_floor,
            "artifact_floor": self.artifact_floor,
            "motion_floor": self.motion_floor,
            "overall_floor": self.overall_floor,
            "multi_character_interaction_floor": self.multi_character_interaction_floor,
            "anatomy_floor": self.anatomy_floor,
            "locomotion_floor": self.locomotion_floor,
            "object_interaction_floor": self.object_interaction_floor,
            "camera_motion_floor": self.camera_motion_floor,
            "lighting_transition_floor": self.lighting_transition_floor,
            "physics_plausibility_floor": self.physics_plausibility_floor,
            "dialogue_lip_sync_floor": self.dialogue_lip_sync_floor,
        }
        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

    def snapshot(self) -> dict[str, float | str]:
        return {
            "schema": "cineos-sequence-quality-policy/0.3",
            "identity_floor": self.identity_floor,
            "temporal_floor": self.temporal_floor,
            "artifact_floor": self.artifact_floor,
            "motion_floor": self.motion_floor,
            "overall_floor": self.overall_floor,
            "multi_character_interaction_floor": self.multi_character_interaction_floor,
            "anatomy_floor": self.anatomy_floor,
            "locomotion_floor": self.locomotion_floor,
            "object_interaction_floor": self.object_interaction_floor,
            "camera_motion_floor": self.camera_motion_floor,
            "lighting_transition_floor": self.lighting_transition_floor,
            "physics_plausibility_floor": self.physics_plausibility_floor,
            "dialogue_lip_sync_floor": self.dialogue_lip_sync_floor,
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


def _shot_challenge_tags(shot: Any) -> frozenset[str]:
    """Return normalized difficult-case declarations without trusting malformed metadata."""

    metadata = getattr(shot, "metadata", None)
    if not isinstance(metadata, Mapping):
        return frozenset()

    tags: set[str] = set()
    for key in _CHALLENGE_METADATA_KEYS:
        raw = metadata.get(key)
        if raw is None:
            continue
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise SequenceQualityError(
                f"shot quality challenge metadata {key!r} must be a sequence"
            )
        for value in raw:
            if not isinstance(value, str) or not value.strip():
                raise SequenceQualityError(
                    f"shot quality challenge metadata {key!r} contains an invalid tag"
                )
            tags.add(value.strip())
    return frozenset(tags)


def _required_challenge_metrics(shot: Any) -> dict[str, str]:
    tags = _shot_challenge_tags(shot)
    return {
        metric_name: challenge
        for challenge, metric_name in CHALLENGE_METRIC_REQUIREMENTS.items()
        if challenge in tags
    }


def _overall_score(metrics: Mapping[str, float]) -> float:
    """Weight identity/temporal evidence above secondary aesthetic metrics."""

    core = (
        0.32 * metrics["identity_similarity"]
        + 0.30 * metrics["temporal_consistency"]
        + 0.20 * metrics["artifact_integrity"]
        + 0.18 * metrics["motion_quality"]
    )
    optional_names = tuple(dict.fromkeys(CHALLENGE_METRIC_REQUIREMENTS.values()))
    optional = [metrics[name] for name in optional_names if name in metrics]
    if not optional:
        return core
    optional_mean = sum(optional) / len(optional)
    return 0.85 * core + 0.15 * optional_mean


def _quality_report(
    metrics: Mapping[str, float],
    policy: SequenceQualityPolicy,
    *,
    required_challenge_metrics: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    required = dict(required_challenge_metrics or {})
    missing_required = sorted(metric for metric in required if metric not in metrics)
    if missing_required:
        details = ", ".join(
            f"{metric} ({required[metric]})" for metric in missing_required
        )
        raise SequenceQualityError(
            "competitive difficult-case quality evidence missing required metric(s): "
            + details
        )

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
        "multi_character_interaction_quality": (
            "repair measured multi-character interaction, contact, turn-taking, and identity separation"
        ),
        "anatomy_quality": "repair hand, finger, limb, and body anatomy before acceptance",
        "locomotion_quality": (
            "repair measured walking/running gait, foot contact, balance, and limb timing"
        ),
        "object_interaction_quality": (
            "repair contact, grip, occlusion, and object interaction fidelity"
        ),
        "camera_motion_quality": (
            "repair measured fast-camera trajectory, framing continuity, and motion coherence"
        ),
        "lighting_transition_quality": (
            "repair measured lighting-transition continuity, exposure response, and scene consistency"
        ),
        "physics_plausibility": (
            "repair measured physical causality, trajectories, collisions, and material response"
        ),
        "dialogue_lip_sync": "improve measured mouth-to-dialogue synchronization",
    }
    for name, threshold in thresholds.items():
        if metrics[name] < threshold:
            failures.append(name)
            directives.append(directive_by_metric[name])

    challenge_thresholds = {
        "multi_character_interaction_quality": policy.multi_character_interaction_floor,
        "anatomy_quality": policy.anatomy_floor,
        "locomotion_quality": policy.locomotion_floor,
        "object_interaction_quality": policy.object_interaction_floor,
        "camera_motion_quality": policy.camera_motion_floor,
        "lighting_transition_quality": policy.lighting_transition_floor,
        "physics_plausibility": policy.physics_plausibility_floor,
        "dialogue_lip_sync": policy.dialogue_lip_sync_floor,
    }
    for name in required:
        threshold = challenge_thresholds[name]
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
        "schema": "cineos-sequence-quality-report/0.3",
        "accepted": accepted,
        "decision": "accept" if accepted else "reject",
        "score": overall,
        "metrics": dict(metrics),
        "failed_metrics": failures,
        "directives": directives,
        "required_challenge_metrics": dict(required),
        "policy": policy.snapshot(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_semantic_scorer_provenance(
    raw: Mapping[str, Any],
) -> dict[str, Any] | None:
    provenance = raw.get("semantic_scorer")
    if provenance is None:
        return None
    if not isinstance(provenance, Mapping):
        raise SequenceQualityError(
            "production quality semantic_scorer provenance must be a mapping"
        )
    normalized = dict(provenance)
    schema = normalized.get("schema")
    origin = normalized.get("origin")
    if not isinstance(schema, str) or not schema.strip():
        raise SequenceQualityError(
            "production quality semantic_scorer provenance requires a non-empty schema"
        )
    if not isinstance(origin, str) or not origin.strip():
        raise SequenceQualityError(
            "production quality semantic_scorer provenance requires a non-empty origin"
        )
    return normalized


class CineosSequenceQualityEvaluator:
    """Callable bridge from measured shot metrics to the rerender loop."""

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
        return _quality_report(
            metrics,
            self.policy,
            required_challenge_metrics=_required_challenge_metrics(shot),
        )


class ArtifactMeasuredSequenceQualityEvaluator:
    """Production evaluator requiring attested metrics bound to the rendered video."""

    production_measurement_evidence = True

    def __init__(
        self, metric_extractor: Any, policy: SequenceQualityPolicy | None = None
    ) -> None:
        if not callable(metric_extractor):
            raise TypeError("metric_extractor must be callable")
        if (
            getattr(metric_extractor, "production_measurement_evidence", False)
            is not True
        ):
            raise TypeError(
                "production metric extractor must attest production_measurement_evidence=True"
            )
        observer_id = getattr(metric_extractor, "observer_id", None)
        if not isinstance(observer_id, str) or not observer_id.strip():
            raise TypeError(
                "production metric extractor must expose a non-empty observer_id"
            )
        self.metric_extractor = metric_extractor
        self.observer_id = observer_id.strip()
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
        if raw.get("production_measurement_evidence") is not True:
            raise SequenceQualityError(
                "production quality measurement must attest "
                "production_measurement_evidence=True"
            )
        observer_id = raw.get("observer_id")
        if not isinstance(observer_id, str) or not observer_id.strip():
            raise SequenceQualityError(
                "production quality measurement requires a non-empty observer_id"
            )
        if observer_id.strip() != self.observer_id:
            raise SequenceQualityError(
                "production quality measurement observer_id does not match attested observer"
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
        semantic_scorer = _validated_semantic_scorer_provenance(raw)
        metrics = _validated_metrics(raw_metrics)
        report = _quality_report(
            metrics,
            self.policy,
            required_challenge_metrics=_required_challenge_metrics(shot),
        )
        report["production_measurement_evidence"] = True
        measurement: dict[str, Any] = {
            "schema": PRODUCTION_MEASUREMENT_SCHEMA,
            "observer_id": self.observer_id,
            "artifact_sha256": artifact_sha256,
            "observer_attested": True,
            "measurement_attested": True,
        }
        if semantic_scorer is not None:
            measurement["semantic_scorer"] = semantic_scorer
        report["measurement"] = measurement
        return report


__all__ = [
    "CHALLENGE_METRIC_REQUIREMENTS",
    "CORE_METRICS",
    "PRODUCTION_MEASUREMENT_SCHEMA",
    "ArtifactMeasuredSequenceQualityEvaluator",
    "CineosSequenceQualityEvaluator",
    "SequenceQualityError",
    "SequenceQualityPolicy",
]
