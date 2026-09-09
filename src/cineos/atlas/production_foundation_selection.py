"""Quality-first selection of pinned external video foundations for production.

CINEOS owns this selection policy, not the pretrained weights. The selector always
tries the strongest approved pinned profile first and falls back only when observed
GPU capacity and the exact native shot contract can safely execute it. Selection is
evidence about runtime fit, not a claim that either external foundation is
CINEOS-native.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .foundation_profiles import (
    WAN22_I2V_A14B_PROFILE,
    WAN22_TI2V_5B_PROFILE,
    FoundationExecutionProfile,
)
from .gpu_preflight import (
    GPUDeviceProfile,
    GPUExecutionPlan,
    GPUPreflightError,
    select_gpu_execution,
)
from .native_request import NativeShotRequest


class ProductionFoundationSelectionError(RuntimeError):
    """Raised when no approved pinned production foundation can safely execute."""


@dataclass(frozen=True, slots=True)
class ProductionFoundationSelection:
    """Auditable quality-first foundation choice and its observed execution plan."""

    profile: FoundationExecutionProfile
    plan: GPUExecutionPlan
    fallback_used: bool
    rejected_profiles: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        provenance = self.profile.provenance
        return {
            "profile_id": self.profile.profile_id,
            "origin": self.profile.origin,
            "model_id": provenance.model_id,
            "revision": provenance.revision,
            "license_id": provenance.license_id,
            "source_url": provenance.source_url,
            "foundation_name": provenance.foundation_name,
            "fallback_used": self.fallback_used,
            "rejected_profiles": list(self.rejected_profiles),
            "execution_plan": {
                "device": self.plan.device,
                "dtype": self.plan.dtype,
                "memory_strategy": self.plan.memory_strategy,
                "enable_vae_tiling": self.plan.enable_vae_tiling,
                "enable_vae_slicing": self.plan.enable_vae_slicing,
                "enable_attention_slicing": self.plan.enable_attention_slicing,
                "estimated_model_vram_gb": self.plan.estimated_model_vram_gb,
                "observed_total_vram_gb": self.plan.observed_total_vram_gb,
                "observed_free_vram_gb": self.plan.observed_free_vram_gb,
                "fit_margin_gb": self.plan.fit_margin_gb,
            },
        }


def _all_shots_are_image_conditioned(requests: Sequence[NativeShotRequest]) -> bool:
    return bool(requests) and all(
        request.approved_reference_ids for request in requests
    )


def _usable_vram_gb(device: GPUDeviceProfile) -> float:
    return (
        device.free_vram_gb if device.free_vram_gb is not None else device.total_vram_gb
    )


def _has_declared_vram_floor(
    devices: tuple[GPUDeviceProfile, ...], profile: FoundationExecutionProfile
) -> bool:
    """Honor a profile's declared production floor before considering offload."""

    return any(
        _usable_vram_gb(device) >= profile.minimum_gpu_vram_gb for device in devices
    )


def _camera_contract(
    request: NativeShotRequest,
) -> tuple[tuple[int, int], float, float]:
    """Read the exact values consumed by the Diffusers renderer.

    The defaults intentionally mirror ``DiffusersVideoRenderer.render``. Keeping the
    production selector aligned with the execution boundary prevents a profile from
    being selected on metadata that the renderer will not actually consume.
    """

    camera: Any = getattr(request, "camera", None)
    if not isinstance(camera, dict):
        raise ValueError("camera must be a mapping")
    raw_resolution = camera.get("resolution", (832, 480))
    if (
        not isinstance(raw_resolution, (list, tuple))
        or len(raw_resolution) != 2
        or isinstance(raw_resolution[0], bool)
        or isinstance(raw_resolution[1], bool)
    ):
        raise ValueError("camera.resolution must contain width and height")
    resolution = (int(raw_resolution[0]), int(raw_resolution[1]))
    fps = float(camera.get("fps", 24.0))
    duration = float(camera.get("duration", 5.0))
    if resolution[0] <= 0 or resolution[1] <= 0 or fps <= 0 or duration <= 0:
        raise ValueError("camera resolution, fps and duration must be positive")
    return resolution, fps, duration


def _renderer_requirement_mismatch(
    request: NativeShotRequest,
    *,
    resolution: tuple[int, int],
    fps: float,
    duration: float,
) -> str | None:
    """Reject contradictory native capability metadata before model acquisition.

    ``DiffusersVideoRenderer`` consumes the camera contract above. Native requests
    compiled from a ``ConditioningPackage`` also carry renderer capability metadata.
    Those fields are useful for routing and evidence, but must never disagree with
    the values that are actually rendered. Otherwise a request could claim one FPS,
    resolution or duration envelope while the GPU renderer silently executes another.

    Missing fields remain backwards compatible for hand-authored legacy requests.
    When present, resolution/FPS are exact requirements and ``maximum_duration`` is
    an upper bound, matching ``RendererCapabilityRequirements`` semantics.
    """

    requirements: Any = getattr(request, "renderer_requirements", None)
    if requirements in (None, {}):
        return None
    if not isinstance(requirements, dict):
        return "renderer_requirements must be a mapping"

    raw_resolution = requirements.get("supported_resolution")
    if raw_resolution is not None:
        if (
            not isinstance(raw_resolution, (list, tuple))
            or len(raw_resolution) != 2
            or isinstance(raw_resolution[0], bool)
            or isinstance(raw_resolution[1], bool)
        ):
            return "renderer_requirements.supported_resolution is malformed"
        required_resolution = (int(raw_resolution[0]), int(raw_resolution[1]))
        if required_resolution != resolution:
            return (
                "camera resolution conflicts with renderer_requirements: "
                f"camera={resolution[0]}x{resolution[1]} "
                f"required={required_resolution[0]}x{required_resolution[1]}"
            )

    raw_fps = requirements.get("supported_fps")
    if raw_fps is not None:
        if isinstance(raw_fps, bool):
            return "renderer_requirements.supported_fps is malformed"
        required_fps = float(raw_fps)
        if required_fps <= 0:
            return "renderer_requirements.supported_fps must be positive"
        if required_fps != fps:
            return (
                "camera fps conflicts with renderer_requirements: "
                f"camera={fps:g} required={required_fps:g}"
            )

    raw_maximum_duration = requirements.get("maximum_duration")
    if raw_maximum_duration is not None:
        if isinstance(raw_maximum_duration, bool):
            return "renderer_requirements.maximum_duration is malformed"
        maximum_duration = float(raw_maximum_duration)
        if maximum_duration <= 0:
            return "renderer_requirements.maximum_duration must be positive"
        if duration > maximum_duration:
            return (
                "camera duration exceeds renderer_requirements.maximum_duration: "
                f"camera={duration:g}s maximum={maximum_duration:g}s"
            )
    return None


def _profile_request_mismatch(
    profile: FoundationExecutionProfile,
    requests: Sequence[NativeShotRequest],
) -> str | None:
    """Return why the exact native requests cannot execute on ``profile``.

    Foundation selection must account for generation resolution, FPS, duration and
    T2V/I2V mode. Otherwise a high-VRAM runner can select a stronger checkpoint whose
    pinned execution profile cannot legally execute the supplied shot contract. That
    would fail only after expensive model setup and, worse, could tempt callers to
    silently reinterpret cinematic timing or resolution to hit a milestone date.
    """

    for index, request in enumerate(requests):
        try:
            resolution, fps, duration = _camera_contract(request)
            requirement_mismatch = _renderer_requirement_mismatch(
                request,
                resolution=resolution,
                fps=fps,
                duration=duration,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return f"shot {index} has invalid renderer camera contract: {exc}"
        if requirement_mismatch is not None:
            return f"shot {index} has contradictory native render contract: {requirement_mismatch}"

        if resolution not in profile.resolutions:
            return (
                f"shot {index} requests unsupported resolution "
                f"{resolution[0]}x{resolution[1]}"
            )
        if fps not in profile.fps:
            return f"shot {index} requests unsupported fps {fps:g}"
        minimum_duration, maximum_duration = profile.duration_range
        if not minimum_duration <= duration <= maximum_duration:
            return f"shot {index} requests unsupported duration {duration:g}s"

        approved_refs = getattr(request, "approved_reference_ids", ())
        required_feature = "image_to_video" if approved_refs else "text_to_video"
        if required_feature not in profile.supported_features:
            return f"shot {index} requires unsupported feature {required_feature}"
    return None


def _execution_plan_mismatch(
    profile: FoundationExecutionProfile,
    plan: GPUExecutionPlan,
) -> str | None:
    """Reject execution modes that are not validated for a production profile.

    The A14B profile's declared 80 GB floor describes the approved resident-quality
    path. The generic GPU planner deliberately supports CPU offload for model-agnostic
    workloads, but CINEOS has not validated A14B offload as equivalent production
    execution. Until a separately benchmarked offload profile exists, accepting one
    here would turn a capacity estimate into an unsupported production claim.
    """

    if profile is WAN22_I2V_A14B_PROFILE and plan.memory_strategy != "resident":
        return (
            "requires validated resident execution; generic planner selected "
            f"{plan.memory_strategy}"
        )
    return None


def select_strongest_production_foundation(
    devices: tuple[GPUDeviceProfile, ...],
    requests: Sequence[NativeShotRequest],
) -> ProductionFoundationSelection:
    """Choose the strongest approved pinned foundation the runner can safely fit.

    Wan2.2 I2V A14B is preferred because it is the repository's quality-first profile,
    but it is only eligible when every shot has approved image conditioning and the
    observed runner meets its declared 80 GB production floor. Each candidate must
    also support the *exact* generation resolution/FPS/duration and T2V/I2V mode that
    the renderer will consume. CINEOS does not silently rewrite those timing or image
    contracts just to make a stronger checkpoint appear executable.

    The generic GPU planner may use offload below a model estimate, but doing that for
    A14B has not yet been production-validated and therefore must not silently weaken
    this quality profile. The pinned Wan2.2 TI2V 5B profile remains the transparent
    lower-memory fallback only when the same native shot contract is compatible.
    """

    candidates: list[FoundationExecutionProfile] = []
    rejected: list[str] = []

    if not _all_shots_are_image_conditioned(requests):
        rejected.append(
            f"{WAN22_I2V_A14B_PROFILE.profile_id}: requires approved image "
            "conditioning on every shot"
        )
    elif not _has_declared_vram_floor(devices, WAN22_I2V_A14B_PROFILE):
        rejected.append(
            f"{WAN22_I2V_A14B_PROFILE.profile_id}: observed usable VRAM is below "
            f"declared {WAN22_I2V_A14B_PROFILE.minimum_gpu_vram_gb:.0f} GB "
            "production floor"
        )
    else:
        candidates.append(WAN22_I2V_A14B_PROFILE)
    candidates.append(WAN22_TI2V_5B_PROFILE)

    for index, profile in enumerate(candidates):
        mismatch = _profile_request_mismatch(profile, requests)
        if mismatch is not None:
            rejected.append(f"{profile.profile_id}: {mismatch}")
            continue
        try:
            plan = select_gpu_execution(
                devices,
                estimated_model_vram_gb=profile.minimum_gpu_vram_gb,
            )
        except GPUPreflightError as exc:
            rejected.append(f"{profile.profile_id}: {exc}")
            continue
        execution_mismatch = _execution_plan_mismatch(profile, plan)
        if execution_mismatch is not None:
            rejected.append(f"{profile.profile_id}: {execution_mismatch}")
            continue
        return ProductionFoundationSelection(
            profile=profile,
            plan=plan,
            fallback_used=(index > 0 or profile is WAN22_TI2V_5B_PROFILE),
            rejected_profiles=tuple(rejected),
        )

    detail = "; ".join(rejected) if rejected else "no approved candidates"
    raise ProductionFoundationSelectionError(
        "no approved pinned production video foundation can safely execute: " + detail
    )


__all__ = [
    "ProductionFoundationSelection",
    "ProductionFoundationSelectionError",
    "select_strongest_production_foundation",
]
