"""Production temporal-lattice integration for external Diffusers foundations.

CINEOS owns film timing and continuity orchestration. External pretrained video
foundations may impose a native temporal VAE lattice; this layer translates an
exact CINEOS shot duration into the smallest valid foundation frame count before
inference, then trims only the extra lattice frames before export and continuity
handoff. The borrowed foundation remains explicitly external.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .diffusers_video import DiffusersVideoError, DiffusersVideoRenderer
from .native_request import NativeShotRequest
from .production_continuity_identity import (
    ProductionContinuityIdentityDiffusersVideoRenderer,
)
from .production_diffusers import ProductionDiffusersVideoResult
from .temporal_lattice import (
    TemporalFramePlan,
    TemporalFramePlanError,
    plan_temporal_frames,
    trim_generated_frames,
)

TEMPORAL_FRAME_PLAN_SCHEMA = "cineos-temporal-frame-plan/0.1"


class TemporalLatticeProductionDiffusersVideoRenderer(
    ProductionContinuityIdentityDiffusersVideoRenderer
):
    """Strict production renderer that honors a foundation-native frame lattice.

    The public request and its content hash remain the source of truth. A temporary
    internal request adjusts only camera duration so the existing Diffusers boundary
    asks the external pipeline for the exact foundation-native frame count. Generated
    frames are then validated and trimmed back to the requested film frame count
    before video export. Continuity inherits the final exported frame, never an
    otherwise invisible padding frame.
    """

    def __init__(
        self,
        *args: Any,
        temporal_compression_ratio: int | None = None,
        max_generation_frames: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_generation_frames = max_generation_frames
        self._active_temporal_plan: TemporalFramePlan | None = None

    def render(self, request: Any) -> ProductionDiffusersVideoResult:
        if not isinstance(request, NativeShotRequest):
            return super().render(request)
        if self.temporal_compression_ratio is None:
            return super().render(request)

        camera = dict(request.camera)
        try:
            fps = float(camera.get("fps", 24.0))
            duration = float(camera.get("duration", 5.0))
            plan = plan_temporal_frames(
                duration,
                fps,
                temporal_compression_ratio=self.temporal_compression_ratio,
                max_generation_frames=self.max_generation_frames,
            )
        except (TypeError, ValueError, TemporalFramePlanError) as exc:
            raise DiffusersVideoError(
                f"invalid production temporal frame plan: {exc}"
            ) from exc

        camera["duration"] = plan.generated_frames / plan.fps
        inference_request = replace(request, camera=camera)
        self._active_temporal_plan = plan
        try:
            result = super().render(inference_request)
        finally:
            self._active_temporal_plan = None

        provenance = dict(result.conditioning_provenance or {})
        provenance["temporal_frame_plan"] = {
            "schema": TEMPORAL_FRAME_PLAN_SCHEMA,
            "requested_frames": plan.requested_frames,
            "foundation_generated_frames": plan.generated_frames,
            "export_frames": plan.export_frames,
            "fps": plan.fps,
            "requested_duration_seconds": plan.duration_seconds,
            "temporal_compression_ratio": plan.temporal_compression_ratio,
            "trimmed_frames": plan.generated_frames - plan.export_frames,
        }
        self._last_conditioning_provenance = dict(provenance)
        return replace(
            result,
            frame_count=plan.export_frames,
            request_hash=request.content_hash,
            conditioning_provenance=provenance,
        )

    def _extract_frames(self, output: Any) -> list[Any]:
        plan = self._active_temporal_plan
        if plan is None:
            return super()._extract_frames(output)

        frames = DiffusersVideoRenderer._extract_frames(output)
        try:
            exported = trim_generated_frames(frames, plan)
        except TemporalFramePlanError as exc:
            raise DiffusersVideoError(
                f"foundation violated the production temporal frame plan: {exc}"
            ) from exc
        self._captured_terminal_frame = exported[-1]
        return exported


__all__ = [
    "TEMPORAL_FRAME_PLAN_SCHEMA",
    "TemporalLatticeProductionDiffusersVideoRenderer",
]
