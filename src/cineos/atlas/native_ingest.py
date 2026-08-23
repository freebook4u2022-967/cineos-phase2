"""Atlas ingestion boundary for CINEOS native shot requests."""

from __future__ import annotations

from dataclasses import dataclass
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


def ingest_native_request(
    session: RendererSession, request: NativeShotRequest
) -> NativeRenderReceipt:
    """Validate, negotiate, and dispatch one native shot request to Atlas."""
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
    return NativeRenderReceipt(
        shot_id=request.shot_id,
        scene_id=request.scene_id,
        request_hash=request.content_hash,
        result=result,
    )
