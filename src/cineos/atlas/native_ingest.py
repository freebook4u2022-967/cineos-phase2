"""Atlas ingestion boundary for CINEOS native shot requests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .capabilities import CapabilityError
from .native_request import NATIVE_SHOT_SCHEMA, NativeShotRequest
from .session import RendererSession


class NativeRequestError(ValueError):
    """Raised when a native shot request is invalid or incompatible."""


@dataclass(slots=True)
class NativeRenderReceipt:
    shot_id: str
    scene_id: str
    request_hash: str
    result: Any
    artifact_path: str | None = None
    artifact_bytes: int | None = None
    artifact_sha256: str | None = None


def _required_features(request: NativeShotRequest) -> tuple[str, ...]:
    requirements = request.renderer_requirements
    features: list[str] = []
    mapping = {
        "image_reference_support": "image-reference",
        "multi_reference_support": "multi-reference",
        "face_identity_support": "face-identity",
        "control_image_support": "control-image",
        "motion_reference_support": "motion-reference",
    }
    for key, feature in mapping.items():
        if requirements.get(key):
            features.append(feature)
    performance = request.performance.get("capability_requirements", {})
    if isinstance(performance, dict):
        features.extend(str(key) for key, value in performance.items() if value)
    return tuple(sorted(set(features)))


def validate_native_request(request: NativeShotRequest) -> None:
    """Validate schema, identity references and deterministic integrity."""
    if not isinstance(request, NativeShotRequest):
        raise TypeError("request must be a NativeShotRequest")
    if request.schema != NATIVE_SHOT_SCHEMA:
        raise NativeRequestError(f"unsupported native request schema: {request.schema}")
    if not request.shot_id or not request.scene_id:
        raise NativeRequestError("native request requires shot_id and scene_id")
    if not request.approved_reference_ids:
        raise NativeRequestError("native request requires approved references")
    original_hash = request.content_hash
    computed_hash = request.refresh_hash()
    if original_hash and original_hash != computed_hash:
        request.content_hash = original_hash
        raise NativeRequestError("native request content hash mismatch")

    approved = set(request.approved_reference_ids)
    for character in request.characters:
        refs = set(character.get("approved_reference_ids", []))
        if not refs:
            raise NativeRequestError(
                "character conditioning has no approved references"
            )
        if not refs.issubset(approved):
            raise NativeRequestError(
                "character conditioning references are outside request approval set"
            )


def _claimed_output_path(result: Any) -> str | Path | None:
    """Return an artifact path only when a renderer explicitly claims one."""
    if isinstance(result, dict):
        return result.get("output_path")
    return getattr(result, "output_path", None)


def _verify_claimed_artifact(result: Any) -> tuple[str | None, int | None, str | None]:
    """Fail closed when a renderer claims a video artifact that is not durable.

    Lightweight/non-file renderers remain backwards compatible because they may
    return arbitrary in-memory results without ``output_path``. Once a renderer
    claims an artifact path, however, Atlas requires a real non-empty file and
    records content-addressed evidence in the receipt. This prevents an exporter
    stub, failed encoder, or stale path from being mistaken for real generated
    footage in production and benchmark workflows.
    """
    claimed = _claimed_output_path(result)
    if claimed is None:
        return None, None, None

    artifact = Path(claimed)
    if not artifact.is_file():
        raise NativeRequestError(
            f"renderer returned missing output artifact: {artifact}"
        )
    byte_size = artifact.stat().st_size
    if byte_size <= 0:
        raise NativeRequestError(f"renderer returned empty output artifact: {artifact}")

    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return str(artifact), byte_size, digest.hexdigest()


def ingest_native_request(
    session: RendererSession, request: NativeShotRequest
) -> NativeRenderReceipt:
    """Validate, negotiate, dispatch, and verify one native Atlas render request."""
    if not isinstance(session, RendererSession):
        raise TypeError("session must be a RendererSession")
    validate_native_request(request)
    camera = request.camera
    requirements = request.renderer_requirements
    character_count = int(requirements.get("character_count", len(request.characters)))
    maximum = session.capabilities.maximum_character_count
    if maximum is not None and character_count > maximum:
        raise CapabilityError(
            f"unsupported character count {character_count}; maximum is {maximum}"
        )

    resolution = tuple(camera.get("resolution", (1920, 1080)))
    duration = float(camera.get("duration", 0.0))
    fps = float(camera.get("fps", 24.0))
    session.negotiate(
        resolution=resolution,
        duration=duration,
        fps=fps,
        features=_required_features(request),
    )
    result = session.render(request)
    artifact_path, artifact_bytes, artifact_sha256 = _verify_claimed_artifact(result)
    return NativeRenderReceipt(
        shot_id=request.shot_id,
        scene_id=request.scene_id,
        request_hash=request.content_hash,
        result=result,
        artifact_path=artifact_path,
        artifact_bytes=artifact_bytes,
        artifact_sha256=artifact_sha256,
    )
