"""Artifact-bound cross-shot continuity evidence for production film QC.

Visual handoff conditions a continuation on its predecessor, but conditioning is
not evidence that the transition actually remained coherent. This module validates
measured predecessor-terminal/current-initial seam evidence and binds it to the
exact accepted artifacts. The scorer remains an external measurement component;
this validator is CINEOS-owned evidence policy and does not relabel scorer or
foundation capability as native model weights.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRANSITION_QUALITY_SCHEMA = "cineos-transition-quality-measurement/0.1"


class TransitionQualityError(ValueError):
    """Raised when cross-shot continuity evidence is incomplete or substituted."""


@dataclass(frozen=True, slots=True)
class TransitionQualityPolicy:
    """Versioned floors for measured visual seam acceptance."""

    visual_similarity_floor: float = 0.78
    motion_boundary_floor: float = 0.72

    def __post_init__(self) -> None:
        for name, value in (
            ("visual_similarity_floor", self.visual_similarity_floor),
            ("motion_boundary_floor", self.motion_boundary_floor),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TransitionQualityError(f"transition evidence requires {field}")
    return value.strip()


def validate_transition_quality_evidence(
    report: Mapping[str, Any],
    *,
    previous_receipt: Any,
    current_receipt: Any,
    current_request: Any,
) -> dict[str, Any]:
    """Validate measured seam evidence against exact predecessor/current artifacts."""

    if not isinstance(report, Mapping):
        raise TransitionQualityError("transition quality report must be a mapping")
    if report.get("schema") != TRANSITION_QUALITY_SCHEMA:
        raise TransitionQualityError("unsupported transition quality evidence schema")
    if report.get("production_measurement_evidence") is not True:
        raise TransitionQualityError(
            "transition report is not production measurement evidence"
        )
    if not isinstance(report.get("accepted"), bool):
        raise TransitionQualityError("transition report requires boolean accepted")

    observer_id = _required_text(report.get("observer_id"), field="observer_id")
    previous_sha = _required_text(
        getattr(previous_receipt, "output_sha256", None),
        field="previous receipt output_sha256",
    )
    current_sha = _required_text(
        getattr(current_receipt, "output_sha256", None),
        field="current receipt output_sha256",
    )
    if len(previous_sha) != 64 or len(current_sha) != 64:
        raise TransitionQualityError("receipt artifact hashes must be SHA-256 digests")
    if report.get("previous_output_sha256") != previous_sha:
        raise TransitionQualityError(
            "transition evidence predecessor artifact hash does not match receipt"
        )
    if report.get("current_output_sha256") != current_sha:
        raise TransitionQualityError(
            "transition evidence current artifact hash does not match receipt"
        )

    previous_result = getattr(previous_receipt, "result", None)
    current_result = getattr(current_receipt, "result", None)
    if previous_result is None or current_result is None:
        raise TransitionQualityError("transition receipts must contain render results")

    previous_scene = getattr(previous_result, "scene_id", None)
    previous_shot = getattr(previous_result, "shot_id", None)
    current_scene = getattr(current_result, "scene_id", None)
    current_shot = getattr(current_result, "shot_id", None)
    expected_identity = {
        "previous_scene_id": previous_scene,
        "previous_shot_id": previous_shot,
        "current_scene_id": current_scene,
        "current_shot_id": current_shot,
    }
    for field, expected in expected_identity.items():
        if report.get(field) != expected:
            raise TransitionQualityError(
                f"transition evidence {field} does not match render lineage"
            )

    if getattr(current_request, "scene_id", None) != current_scene:
        raise TransitionQualityError("current request scene does not match receipt")
    if getattr(current_request, "shot_id", None) != current_shot:
        raise TransitionQualityError("current request shot does not match receipt")
    continuity = getattr(current_request, "continuity", None)
    if not isinstance(continuity, Mapping):
        raise TransitionQualityError("current request has no continuity mapping")
    declared = continuity.get("previous_shot")
    legacy = continuity.get("previous_shot_id")
    if declared is not None and legacy is not None and declared != legacy:
        raise TransitionQualityError(
            "current request has conflicting predecessor links"
        )
    linked_shot = declared if declared is not None else legacy
    if linked_shot != previous_shot:
        raise TransitionQualityError(
            "current request predecessor does not match accepted previous artifact"
        )
    linked_scene = continuity.get("previous_scene_id", current_scene)
    if linked_scene != previous_scene:
        raise TransitionQualityError(
            "current request predecessor scene does not match accepted artifact"
        )

    sample_count = report.get("measured_sample_count")
    if not isinstance(sample_count, int) or isinstance(sample_count, bool):
        raise TransitionQualityError(
            "transition evidence measured_sample_count must be an integer"
        )
    if sample_count <= 0:
        raise TransitionQualityError(
            "transition evidence requires at least one measured sample"
        )

    normalized = dict(report)
    normalized["observer_id"] = observer_id
    normalized["previous_output_sha256"] = previous_sha
    normalized["current_output_sha256"] = current_sha
    return normalized


class ArtifactMeasuredTransitionQualityEvaluator:
    """Strict adapter for a real two-artifact transition measurement observer.

    The observer must attest itself, expose a stable id, inspect both video files,
    and return hashes plus normalized seam metrics. This prevents a plain injected
    lambda from promoting synthetic scores to production continuity evidence.
    """

    production_measurement_evidence = True

    def __init__(
        self,
        observer: Any,
        policy: TransitionQualityPolicy | None = None,
    ) -> None:
        if not callable(observer):
            raise TypeError("transition observer must be callable")
        if getattr(observer, "production_measurement_evidence", False) is not True:
            raise TypeError(
                "production transition observer must attest measurement evidence"
            )
        observer_id = getattr(observer, "observer_id", None)
        if not isinstance(observer_id, str) or not observer_id.strip():
            raise TypeError("production transition observer requires observer_id")
        self.observer = observer
        self.observer_id = observer_id.strip()
        self.policy = policy or TransitionQualityPolicy()

    def __call__(
        self,
        previous_path: str,
        current_path: str,
        *,
        previous_shot: Any,
        current_shot: Any,
        attempt_index: int,
    ) -> dict[str, Any]:
        previous = Path(previous_path)
        current = Path(current_path)
        if not previous.is_file() or not current.is_file():
            raise TransitionQualityError(
                "production transition observer requires both rendered artifacts"
            )
        previous_sha = _sha256_file(previous)
        current_sha = _sha256_file(current)
        raw = self.observer(
            previous_path,
            current_path,
            previous_shot=previous_shot,
            current_shot=current_shot,
            attempt_index=attempt_index,
        )
        if not isinstance(raw, Mapping):
            raise TransitionQualityError("transition observer must return a mapping")
        if raw.get("schema") != TRANSITION_QUALITY_SCHEMA:
            raise TransitionQualityError("unsupported transition observer schema")
        if raw.get("observer_id") != self.observer_id:
            raise TransitionQualityError("transition observer id changed during measurement")
        if raw.get("previous_output_sha256") != previous_sha:
            raise TransitionQualityError("transition observer predecessor hash mismatch")
        if raw.get("current_output_sha256") != current_sha:
            raise TransitionQualityError("transition observer current hash mismatch")
        sample_count = raw.get("measured_sample_count")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool):
            raise TransitionQualityError("transition observer sample count must be integer")
        if sample_count <= 0:
            raise TransitionQualityError("transition observer measured no samples")

        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping):
            raise TransitionQualityError("transition observer requires metrics mapping")
        visual = metrics.get("visual_seam_similarity")
        motion = metrics.get("motion_boundary_consistency")
        for name, value in (
            ("visual_seam_similarity", visual),
            ("motion_boundary_consistency", motion),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TransitionQualityError(f"transition metric {name} must be numeric")
            if not 0.0 <= float(value) <= 1.0:
                raise TransitionQualityError(f"transition metric {name} out of range")

        failures: list[str] = []
        directives: list[str] = []
        if float(visual) < self.policy.visual_similarity_floor:
            failures.append("visual_seam_similarity")
            directives.append(
                "preserve predecessor composition, lighting, pose, and appearance"
            )
        if float(motion) < self.policy.motion_boundary_floor:
            failures.append("motion_boundary_consistency")
            directives.append("preserve physically coherent motion across the shot boundary")

        return {
            "schema": TRANSITION_QUALITY_SCHEMA,
            "production_measurement_evidence": True,
            "accepted": not failures,
            "observer_id": self.observer_id,
            "previous_scene_id": getattr(previous_shot, "scene_id", None),
            "previous_shot_id": getattr(previous_shot, "shot_id", None),
            "current_scene_id": getattr(current_shot, "scene_id", None),
            "current_shot_id": getattr(current_shot, "shot_id", None),
            "previous_output_sha256": previous_sha,
            "current_output_sha256": current_sha,
            "measured_sample_count": sample_count,
            "metrics": {
                "visual_seam_similarity": float(visual),
                "motion_boundary_consistency": float(motion),
            },
            "failed_metrics": failures,
            "directives": directives,
        }


__all__ = [
    "TRANSITION_QUALITY_SCHEMA",
    "ArtifactMeasuredTransitionQualityEvaluator",
    "TransitionQualityError",
    "TransitionQualityPolicy",
    "validate_transition_quality_evidence",
]
