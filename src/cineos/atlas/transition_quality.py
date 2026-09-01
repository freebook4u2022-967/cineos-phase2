"""Artifact-bound cross-shot continuity evidence for production film QC.

Visual handoff conditions a continuation on its predecessor, but conditioning is
not evidence that the transition actually remained coherent. This module validates
measured predecessor-terminal/current-initial seam evidence and binds it to the
exact accepted artifacts. The scorer remains an external measurement component;
this validator is CINEOS-owned evidence policy and does not relabel scorer or
foundation capability as native model weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TRANSITION_QUALITY_SCHEMA = "cineos-transition-quality-measurement/0.1"


class TransitionQualityError(ValueError):
    """Raised when cross-shot continuity evidence is incomplete or substituted."""


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
    """Validate measured seam evidence against exact predecessor/current artifacts.

    Production evidence must identify the terminal-to-initial scorer, contain at
    least one measured frame/sample pair, and cryptographically bind both artifacts.
    The current request must explicitly point at the accepted predecessor shot.
    """

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
        raise TransitionQualityError("current request has conflicting predecessor links")
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


__all__ = [
    "TRANSITION_QUALITY_SCHEMA",
    "TransitionQualityError",
    "validate_transition_quality_evidence",
]
