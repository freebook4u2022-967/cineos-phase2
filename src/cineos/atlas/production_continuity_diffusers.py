"""Production visual continuity for persistent Diffusers video sessions.

CINEOS owns this orchestration layer; the pretrained video foundation remains
external. A connected shot may use the terminal generated frame of its declared
predecessor as the next image-to-video anchor. This turns ``previous_shot`` from
prompt-only metadata into an actual visual conditioning signal while preserving an
auditable identity-reference and artifact lineage.

The handoff is deliberately in-memory. The production connected benchmark keeps
one renderer/model session alive across all attempts, so no lossy decode/re-encode
step is needed between shots. A continuation cannot be rendered by a fresh
renderer instance: if the declared predecessor frame or its render binding is
unavailable, execution fails closed instead of silently falling back to text-only
generation.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .diffusers_video import DiffusersVideoError, DiffusersVideoRenderer
from .native_request import NativeShotRequest
from .production_diffusers import (
    ProductionDiffusersVideoRenderer,
    ProductionDiffusersVideoResult,
)

VISUAL_CONTINUITY_SCHEMA = "cineos-visual-continuity-conditioning/0.2"


def _continuity_predecessor(request: NativeShotRequest) -> tuple[str, str] | None:
    """Return the declared predecessor scene/shot identity, if any."""

    continuity = request.continuity
    canonical = continuity.get("previous_shot")
    legacy = continuity.get("previous_shot_id")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise DiffusersVideoError(
            "continuation shot has conflicting previous_shot and previous_shot_id"
        )
    previous_shot = canonical if canonical is not None else legacy
    if previous_shot is None:
        return None
    if not isinstance(previous_shot, str) or not previous_shot.strip():
        raise DiffusersVideoError("previous_shot must be a non-empty string or null")

    previous_scene = continuity.get("previous_scene_id", request.scene_id)
    if not isinstance(previous_scene, str) or not previous_scene.strip():
        raise DiffusersVideoError("previous_scene_id must be a non-empty string")
    return previous_scene.strip(), previous_shot.strip()


class ProductionContinuityDiffusersVideoRenderer(ProductionDiffusersVideoRenderer):
    """Strict production renderer with predecessor-frame visual handoff.

    The first shot follows the normal approved-reference path. Every connected
    continuation then consumes the terminal frame captured from its declared
    predecessor. The current shot must declare the exact same approved reference
    lineage as that predecessor; changing identities mid-chain requires an explicit
    new root shot rather than an implicit substitution.

    Each cached terminal frame is also bound to the exact rendered predecessor
    artifact SHA-256 and request hash. Those bindings are copied into the returned
    production result so downstream GPU receipts retain auditable continuity
    evidence without consulting mutable renderer side state.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._terminal_frames: dict[tuple[str, str], Any] = {}
        self._identity_lineage: dict[tuple[str, str], tuple[str, ...]] = {}
        self._render_bindings: dict[tuple[str, str], tuple[str, str]] = {}
        self._active_continuity_frame: Any | None = None
        self._active_predecessor: tuple[str, str] | None = None
        self._active_predecessor_binding: tuple[str, str] | None = None
        self._captured_terminal_frame: Any | None = None
        self._last_conditioning_provenance: dict[str, Any] | None = None

    @property
    def last_conditioning_provenance(self) -> Mapping[str, Any] | None:
        if self._last_conditioning_provenance is None:
            return None
        return dict(self._last_conditioning_provenance)

    def render(self, request: Any) -> ProductionDiffusersVideoResult:
        if not isinstance(request, NativeShotRequest):
            return super().render(request)

        predecessor = _continuity_predecessor(request)
        self._active_continuity_frame = None
        self._active_predecessor = predecessor
        self._active_predecessor_binding = None
        self._captured_terminal_frame = None

        if predecessor is not None:
            if predecessor not in self._terminal_frames:
                raise DiffusersVideoError(
                    "visual continuity predecessor frame is unavailable in the current "
                    f"persistent renderer session: {predecessor[0]}/{predecessor[1]}"
                )
            if predecessor not in self._render_bindings:
                raise DiffusersVideoError(
                    "visual continuity predecessor render binding is unavailable in the "
                    f"current persistent renderer session: {predecessor[0]}/"
                    f"{predecessor[1]}"
                )
            expected_lineage = self._identity_lineage.get(predecessor)
            if expected_lineage is None:
                raise DiffusersVideoError(
                    "visual continuity predecessor identity lineage is unavailable in "
                    "the current persistent renderer session"
                )
            current_lineage = tuple(request.approved_reference_ids)
            if current_lineage != expected_lineage:
                raise DiffusersVideoError(
                    "continuation identity references differ from predecessor lineage; "
                    "start a new root shot or preserve the exact approved reference ids"
                )
            self._active_continuity_frame = self._terminal_frames[predecessor]
            self._active_predecessor_binding = self._render_bindings[predecessor]

        try:
            result = super().render(request)
            if self._captured_terminal_frame is None:
                raise DiffusersVideoError(
                    "foundation output did not expose a terminal frame for continuity"
                )
            if not result.artifact_sha256 or not result.request_hash:
                raise DiffusersVideoError(
                    "production continuity requires artifact and request hashes from "
                    "the completed render"
                )

            identity = (request.scene_id, request.shot_id)
            self._terminal_frames[identity] = self._captured_terminal_frame
            self._identity_lineage[identity] = tuple(request.approved_reference_ids)
            self._render_bindings[identity] = (
                result.artifact_sha256,
                result.request_hash,
            )

            if predecessor is None:
                mode = (
                    "approved_reference_root"
                    if request.approved_reference_ids
                    else "text_only_root"
                )
                predecessor_artifact_sha256 = None
                predecessor_request_hash = None
            else:
                mode = "predecessor_terminal_frame_lineage"
                if self._active_predecessor_binding is None:
                    raise DiffusersVideoError(
                        "visual continuity predecessor render binding disappeared "
                        "during inference"
                    )
                predecessor_artifact_sha256, predecessor_request_hash = (
                    self._active_predecessor_binding
                )

            generic_provenance = result.conditioning_provenance
            continuity_provenance: dict[str, Any] = {
                "schema": VISUAL_CONTINUITY_SCHEMA,
                "mode": mode,
                "scene_id": request.scene_id,
                "shot_id": request.shot_id,
                "previous_scene_id": predecessor[0] if predecessor else None,
                "previous_shot_id": predecessor[1] if predecessor else None,
                "predecessor_artifact_sha256": predecessor_artifact_sha256,
                "predecessor_request_hash": predecessor_request_hash,
                "current_artifact_sha256": result.artifact_sha256,
                "current_request_hash": result.request_hash,
                "approved_reference_ids": list(request.approved_reference_ids),
                "in_memory_terminal_frame": predecessor is not None,
            }
            if generic_provenance is not None:
                continuity_provenance["identity_conditioning"] = dict(
                    generic_provenance
                )

            self._last_conditioning_provenance = continuity_provenance
            return replace(
                result,
                conditioning_provenance=dict(continuity_provenance),
            )
        finally:
            self._active_continuity_frame = None
            self._active_predecessor = None
            self._active_predecessor_binding = None
            self._captured_terminal_frame = None

    def _verify_reference_conditioning_path(self, request: NativeShotRequest) -> None:
        if self._active_continuity_frame is None:
            return super()._verify_reference_conditioning_path(request)
        if self._pipeline is None:
            raise DiffusersVideoError("renderer model is not loaded")
        parameters = inspect.signature(self._pipeline.__call__).parameters
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if "image" not in parameters and not accepts_kwargs:
            raise DiffusersVideoError(
                "connected production shot requires image conditioning for visual "
                "continuity, but the loaded foundation pipeline does not expose it"
            )

    def _load_primary_reference(self, request: NativeShotRequest) -> Any | None:
        if self._active_continuity_frame is not None:
            return self._active_continuity_frame
        return super()._load_primary_reference(request)

    def _extract_frames(self, output: Any) -> list[Any]:
        frames = DiffusersVideoRenderer._extract_frames(output)
        self._captured_terminal_frame = frames[-1]
        return frames


__all__ = [
    "ProductionContinuityDiffusersVideoRenderer",
    "VISUAL_CONTINUITY_SCHEMA",
]
