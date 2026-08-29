"""Fail-closed production wrapper for auditable Atlas video renders.

The low-level Diffusers renderer intentionally stays easy to unit-test and can be
used with injected exporters. Production film generation needs a stricter
boundary: a render is not considered successful until a concrete non-empty
artifact exists and a cryptographic evidence sidecar has been written.

This module binds those requirements without relabelling the pretrained
foundation as CINEOS-native capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .diffusers_video import DiffusersVideoRenderer, DiffusersVideoResult
from .native_request import NativeShotRequest
from .render_evidence import RenderEvidence, collect_render_evidence, write_render_evidence


class VerifiedRenderError(RuntimeError):
    """Raised when a nominal render cannot be promoted to production evidence."""


@dataclass(frozen=True, slots=True)
class VerifiedRenderResult:
    """A rendered shot whose artifact and execution provenance are verified."""

    render: DiffusersVideoResult
    evidence: RenderEvidence
    evidence_path: str


def render_verified(
    renderer: DiffusersVideoRenderer,
    request: NativeShotRequest,
) -> VerifiedRenderResult:
    """Render one shot and atomically bind it to verifiable execution evidence.

    The runtime values are read from the renderer after model loading so the
    evidence records the device, dtype and memory strategy actually selected by
    CINEOS rather than trusting duplicated caller metadata.
    """
    result = renderer.render(request)
    artifact = Path(result.output_path)

    if not artifact.is_file():
        raise VerifiedRenderError(
            f"renderer reported success but artifact does not exist: {artifact}"
        )
    if artifact.stat().st_size <= 0:
        raise VerifiedRenderError(
            f"renderer reported success but artifact is empty: {artifact}"
        )

    foundation = result.foundation
    evidence = collect_render_evidence(
        artifact_path=artifact,
        shot_id=result.shot_id,
        scene_id=result.scene_id,
        frame_count=result.frame_count,
        seed=result.seed,
        request_hash=result.request_hash,
        foundation_model_id=foundation.model_id,
        foundation_revision=foundation.revision,
        foundation_license_id=foundation.license_id,
        device=_runtime_value(renderer, "_device"),
        dtype=_runtime_value(renderer, "_dtype_name"),
        memory_strategy=_runtime_value(renderer, "_memory_strategy"),
    )
    evidence_path = write_render_evidence(evidence)
    return VerifiedRenderResult(
        render=result,
        evidence=evidence,
        evidence_path=str(evidence_path),
    )


def _runtime_value(renderer: Any, name: str) -> str:
    value = getattr(renderer, name, None)
    if not isinstance(value, str) or not value:
        raise VerifiedRenderError(f"renderer runtime value {name!r} is unavailable")
    return value
