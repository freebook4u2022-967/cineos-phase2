"""Auditable request adaptation for CINEOS quality-driven rerender attempts.

A rejected render must never be silently retried with untracked prompt or seed
changes. This module converts measured quality-gate directives into a fresh
:class:`NativeShotRequest` whose hash, seed, and retry lineage are explicit.
The resulting request remains renderer-independent and preserves the original
scene/shot identity, approved references, continuity contract, and provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .native_request import NativeShotRequest


class QualityRetryError(ValueError):
    """Raised when a quality report cannot produce a safe rerender request."""


@dataclass(frozen=True, slots=True)
class QualityRetryPolicy:
    """Deterministic policy for adapting a rejected shot into a new attempt."""

    max_attempts: int = 3
    seed_stride: int = 104_729

    def __post_init__(self) -> None:
        if self.max_attempts < 2:
            raise ValueError("max_attempts must allow at least one retry")
        if self.seed_stride <= 0:
            raise ValueError("seed_stride must be positive")


def _normalized_strings(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise QualityRetryError(f"quality report {field} must be a list or tuple")
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def build_quality_retry_request(
    request: NativeShotRequest,
    report: Mapping[str, Any],
    *,
    attempt_index: int,
    policy: QualityRetryPolicy | None = None,
) -> NativeShotRequest:
    """Build a fresh hash-bound request for one measured rerender attempt.

    ``attempt_index`` is one-based: ``1`` means the first retry after the original
    render. Each retry receives a deterministic seed offset and explicit quality
    directives in metadata. Nothing mutates the original request.
    """

    retry_policy = policy or QualityRetryPolicy()
    if attempt_index < 1:
        raise QualityRetryError("attempt_index must be at least 1 for a retry")
    if attempt_index >= retry_policy.max_attempts:
        raise QualityRetryError(
            f"attempt_index {attempt_index} exceeds max_attempts={retry_policy.max_attempts}"
        )
    if report.get("accepted") is True:
        raise QualityRetryError(
            "accepted quality reports must not create retry requests"
        )

    directives = _normalized_strings(report.get("directives"), field="directives")
    failed_metrics = _normalized_strings(
        report.get("failed_metrics"), field="failed_metrics"
    )
    if not directives:
        raise QualityRetryError(
            "rejected quality report must provide correction directives"
        )

    parent_hash = request.content_hash
    if not parent_hash:
        parent_hash = request.refresh_hash()

    metadata = deepcopy(request.metadata)
    existing_directives = metadata.get("quality_directives", [])
    if existing_directives is None:
        existing_directives = []
    if not isinstance(existing_directives, (list, tuple)):
        raise QualityRetryError(
            "request metadata quality_directives must be a list or tuple"
        )

    merged_directives = _normalized_strings(
        [*existing_directives, *directives], field="quality_directives"
    )
    metadata["quality_directives"] = merged_directives
    metadata["quality_retry"] = {
        "schema": "cineos-quality-retry/0.1",
        "attempt_index": attempt_index,
        "parent_request_hash": parent_hash,
        "original_seed": request.deterministic_seed,
        "failed_metrics": failed_metrics,
        "directives": directives,
    }

    retry = NativeShotRequest(
        shot_id=request.shot_id,
        scene_id=request.scene_id,
        camera=deepcopy(request.camera),
        characters=deepcopy(request.characters),
        environment=deepcopy(request.environment),
        wardrobe=deepcopy(request.wardrobe),
        props=deepcopy(request.props),
        continuity=deepcopy(request.continuity),
        performance=deepcopy(request.performance),
        approved_reference_ids=list(request.approved_reference_ids),
        deterministic_seed=request.deterministic_seed
        + retry_policy.seed_stride * attempt_index,
        renderer_requirements=deepcopy(request.renderer_requirements),
        schema=request.schema,
        metadata=metadata,
    )
    retry.refresh_hash()
    if retry.content_hash == parent_hash:
        raise QualityRetryError("retry request must have a fresh content hash")
    return retry


__all__ = [
    "QualityRetryError",
    "QualityRetryPolicy",
    "build_quality_retry_request",
]
