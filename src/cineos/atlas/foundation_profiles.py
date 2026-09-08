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
from .production_continuity_identity import (
    ProductionContinuityIdentityDiffusersVideoRenderer,
)
from .reference_board import compose_reference_board

EXTERNAL_PRETRAINED_FOUNDATION = "external_pretrained_foundation"
WAN22_TI2V_5B_DIFFUSERS_REVISION = "4c6ca6c2ded5c79550a3ca25555efc561112891a"
WAN22_I2V_A14B_DIFFUSERS_REVISION = "35a0aa3ae258b57ad887d9a0988c736320742096"


@dataclass(frozen=True, slots=True)
class FoundationExecutionProfile:
    """Reproducible execution contract for one external pretrained foundation."""

    profile_id: str
    provenance: FoundationProvenance
    resolutions: tuple[tuple[int, int], ...]
    fps: tuple[float, ...]
    duration_range: tuple[float, float]
    minimum_gpu_vram_gb: float
    supported_features: frozenset[str] = frozenset({"text_to_video", "image_to_video"})
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
        if not self.supported_features:
            raise ValueError("foundation execution profiles require supported features")
        if not self.supported_features.issubset({"text_to_video", "image_to_video"}):
            raise ValueError(
                "foundation execution profile has unsupported feature tags"
            )

    def renderer(
        self,
        *,
        output_dir: str | Path,
        reference_loader: Any | None = None,
        multi_reference_adapter: Any | None = None,
        continuity_identity_adapter: Any | None = None,
        pipeline_factory: Any | None = None,
        video_exporter: Any | None = None,
    ) -> DiffusersVideoRenderer:
        """Build the strict production renderer for this pinned foundation.

        Root multi-character shots use CINEOS' deterministic reference-board adapter
        by default. Connected shots keep the already validated predecessor-terminal-
        frame handoff unless an explicit ``continuity_identity_adapter`` is supplied.
        The fresh-reference compositor remains an experimental CINEOS conditioning
        strategy until a real GPU A/B benchmark demonstrates that it improves
        identity without degrading temporal continuity or renderer quality. This
        prevents an unmeasured preprocessing change from being promoted merely to
        accelerate the milestone date.
        """
        adapter = (
            compose_reference_board
            if multi_reference_adapter is None
            else multi_reference_adapter
        )
        return ProductionContinuityIdentityDiffusersVideoRenderer(
            self.provenance,
            output_dir=output_dir,
            resolutions=self.resolutions,
            duration_range=self.duration_range,
            fps=self.fps,
            supported_features=self.supported_features,
            reference_loader=reference_loader,
            multi_reference_adapter=adapter,
            continuity_identity_adapter=continuity_identity_adapter,
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
            "supported_features": sorted(self.supported_features),
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

# Quality-first, higher-compute option for identity-conditioned production shots.
# Wan-AI publishes this I2V MoE foundation under Apache-2.0 and documents both
# 480P/720P generation. The 80 GB minimum is intentionally conservative and follows
# the upstream single-GPU requirement for the native A14B execution path; CINEOS may
# later validate lower-memory offload/quantized plans separately without weakening
# this resident-quality profile.
WAN22_I2V_A14B_PROFILE = FoundationExecutionProfile(
    profile_id="wan2.2-i2v-a14b-diffusers-quality-pinned-2026-09",
    provenance=FoundationProvenance(
        model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        revision=WAN22_I2V_A14B_DIFFUSERS_REVISION,
        license_id="Apache-2.0",
        source_url="https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        foundation_name="Wan2.2 I2V A14B",
    ),
    resolutions=((1280, 720), (720, 1280), (832, 480), (480, 832)),
    fps=(16.0,),
    duration_range=(1.0, 5.0),
    minimum_gpu_vram_gb=80.0,
    supported_features=frozenset({"image_to_video"}),
)


def build_wan22_ti2v_5b_renderer(
    *,
    output_dir: str | Path,
    reference_loader: Any | None = None,
    multi_reference_adapter: Any | None = None,
    continuity_identity_adapter: Any | None = None,
    pipeline_factory: Any | None = None,
    video_exporter: Any | None = None,
) -> DiffusersVideoRenderer:
    """Build the pinned Wan2.2 bridge without obscuring foundation provenance."""
    return WAN22_TI2V_5B_PROFILE.renderer(
        output_dir=output_dir,
        reference_loader=reference_loader,
        multi_reference_adapter=multi_reference_adapter,
        continuity_identity_adapter=continuity_identity_adapter,
        pipeline_factory=pipeline_factory,
        video_exporter=video_exporter,
    )


def build_wan22_i2v_a14b_renderer(
    *,
    output_dir: str | Path,
    reference_loader: Any | None = None,
    multi_reference_adapter: Any | None = None,
    continuity_identity_adapter: Any | None = None,
    pipeline_factory: Any | None = None,
    video_exporter: Any | None = None,
) -> DiffusersVideoRenderer:
    """Build the pinned high-capacity Wan2.2 I2V bridge for quality-first runs.

    The checkpoint remains an external Apache-2.0 pretrained foundation. CINEOS
    owns the reference composition, continuity conditioning, QC and orchestration
    around it, not the underlying Wan model weights.
    """
    return WAN22_I2V_A14B_PROFILE.renderer(
        output_dir=output_dir,
        reference_loader=reference_loader,
        multi_reference_adapter=multi_reference_adapter,
        continuity_identity_adapter=continuity_identity_adapter,
        pipeline_factory=pipeline_factory,
        video_exporter=video_exporter,
    )


__all__ = [
    "EXTERNAL_PRETRAINED_FOUNDATION",
    "FoundationExecutionProfile",
    "WAN22_I2V_A14B_DIFFUSERS_REVISION",
    "WAN22_I2V_A14B_PROFILE",
    "WAN22_TI2V_5B_DIFFUSERS_REVISION",
    "WAN22_TI2V_5B_PROFILE",
    "build_wan22_i2v_a14b_renderer",
    "build_wan22_ti2v_5b_renderer",
]
