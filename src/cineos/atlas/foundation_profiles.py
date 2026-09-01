"""Pinned execution profiles for third-party pretrained video foundations.

Profiles in this module are intentionally explicit about ownership and provenance.
They provide a reproducible bridge from CINEOS-owned direction/conditioning to an
external pretrained checkpoint without representing those weights as a CINEOS-native
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .diffusers_video import DiffusersVideoRenderer, FoundationProvenance
from .production_continuity_diffusers import ProductionContinuityDiffusersVideoRenderer

EXTERNAL_PRETRAINED_FOUNDATION = "external_pretrained_foundation"
WAN22_TI2V_5B_DIFFUSERS_REVISION = "4c6ca6c2ded5c79550a3ca25555efc561112891a"


@dataclass(frozen=True, slots=True)
class FoundationExecutionProfile:
    """Reproducible execution contract for one external pretrained foundation."""

    profile_id: str
    provenance: FoundationProvenance
    resolutions: tuple[tuple[int, int], ...]
    fps: tuple[float, ...]
    duration_range: tuple[float, float]
    minimum_gpu_vram_gb: float
    origin: str = EXTERNAL_PRETRAINED_FOUNDATION

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must not be empty")
        if self.origin != EXTERNAL_PRETRAINED_FOUNDATION:
            raise ValueError(
                "Diffusers foundation profiles must remain explicitly external"
            )
        if not self.provenance.model_id.strip():
            raise ValueError("foundation model_id must not be empty")
        revision = self.provenance.revision
        if (
            revision is None
            or len(revision) != 40
            or any(
                character not in "0123456789abcdef" for character in revision.lower()
            )
        ):
            raise ValueError(
                "foundation execution profiles require an immutable 40-character "
                "checkpoint revision"
            )
        if not self.provenance.license_id:
            raise ValueError("foundation execution profiles require license metadata")
        if not self.provenance.source_url:
            raise ValueError("foundation execution profiles require a source URL")
        if not self.resolutions:
            raise ValueError("foundation execution profiles require resolutions")
        if not self.fps or any(value <= 0 for value in self.fps):
            raise ValueError("foundation execution profiles require positive fps")
        minimum, maximum = self.duration_range
        if minimum <= 0 or maximum < minimum:
            raise ValueError("invalid foundation duration range")
        if self.minimum_gpu_vram_gb <= 0:
            raise ValueError("minimum_gpu_vram_gb must be positive")

    def renderer(
        self,
        *,
        output_dir: str | Path,
        reference_loader: Any | None = None,
        multi_reference_adapter: Any | None = None,
        pipeline_factory: Any | None = None,
        video_exporter: Any | None = None,
    ) -> DiffusersVideoRenderer:
        """Build the strict production renderer for this pinned foundation."""
        return ProductionContinuityDiffusersVideoRenderer(
            self.provenance,
            output_dir=output_dir,
            resolutions=self.resolutions,
            duration_range=self.duration_range,
            fps=self.fps,
            supported_features=frozenset({"text_to_video", "image_to_video"}),
            reference_loader=reference_loader,
            multi_reference_adapter=multi_reference_adapter,
            pipeline_factory=pipeline_factory,
            video_exporter=video_exporter,
        )

    def snapshot(self) -> dict[str, Any]:
        """Return audit metadata suitable for benchmark and release receipts."""
        return {
            "profile_id": self.profile_id,
            "origin": self.origin,
            "provenance": self.provenance.to_dict(),
            "resolutions": [list(item) for item in self.resolutions],
            "fps": list(self.fps),
            "duration_range": list(self.duration_range),
            "minimum_gpu_vram_gb": self.minimum_gpu_vram_gb,
        }


WAN22_TI2V_5B_PROFILE = FoundationExecutionProfile(
    profile_id="wan2.2-ti2v-5b-diffusers-pinned-2026-08",
    provenance=FoundationProvenance(
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        revision=WAN22_TI2V_5B_DIFFUSERS_REVISION,
        license_id="Apache-2.0",
        source_url="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        foundation_name="Wan2.2 TI2V 5B",
    ),
    resolutions=((1280, 704), (704, 1280)),
    fps=(24.0,),
    duration_range=(1.0, 5.0),
    minimum_gpu_vram_gb=24.0,
)


def build_wan22_ti2v_5b_renderer(
    *,
    output_dir: str | Path,
    reference_loader: Any | None = None,
    multi_reference_adapter: Any | None = None,
    pipeline_factory: Any | None = None,
    video_exporter: Any | None = None,
) -> DiffusersVideoRenderer:
    """Build the pinned Wan2.2 bridge without obscuring foundation provenance."""
    return WAN22_TI2V_5B_PROFILE.renderer(
        output_dir=output_dir,
        reference_loader=reference_loader,
        multi_reference_adapter=multi_reference_adapter,
        pipeline_factory=pipeline_factory,
        video_exporter=video_exporter,
    )


__all__ = [
    "EXTERNAL_PRETRAINED_FOUNDATION",
    "FoundationExecutionProfile",
    "WAN22_TI2V_5B_DIFFUSERS_REVISION",
    "WAN22_TI2V_5B_PROFILE",
    "build_wan22_ti2v_5b_renderer",
]
