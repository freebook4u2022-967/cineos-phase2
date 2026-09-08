"""Quality-first selection of pinned external video foundations for production.

CINEOS owns this selection policy, not the pretrained weights.  The selector always
tries the strongest approved pinned profile first and falls back only when observed
GPU capacity cannot safely execute it.  Selection is evidence about runtime fit, not
a claim that either external foundation is CINEOS-native.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
        return {
            "profile_id": self.profile.profile_id,
            "origin": self.profile.origin,
            "model_id": self.profile.provenance.model_id,
            "revision": self.profile.provenance.revision,
            "fallback_used": self.fallback_used,
            "rejected_profiles": list(self.rejected_profiles),
            "execution_plan": {
                "device": self.plan.device,
                "dtype": self.plan.dtype,
                "memory_strategy": self.plan.memory_strategy,
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


def select_strongest_production_foundation(
    devices: tuple[GPUDeviceProfile, ...],
    requests: Sequence[NativeShotRequest],
) -> ProductionFoundationSelection:
    """Choose the strongest legally approved pinned foundation the runner can fit.

    Wan2.2 I2V A14B is preferred because it is the repository's quality-first profile,
    but it is only eligible when every shot has approved image conditioning.  The
    pinned Wan2.2 TI2V 5B profile remains the lower-memory fallback.  A fallback is
    never silently relabeled as the stronger profile.
    """

    candidates: list[FoundationExecutionProfile] = []
    rejected: list[str] = []

    if _all_shots_are_image_conditioned(requests):
        candidates.append(WAN22_I2V_A14B_PROFILE)
    else:
        rejected.append(
            f"{WAN22_I2V_A14B_PROFILE.profile_id}: requires approved image "
            "conditioning on every shot"
        )
    candidates.append(WAN22_TI2V_5B_PROFILE)

    for index, profile in enumerate(candidates):
        try:
            plan = select_gpu_execution(
                devices,
                estimated_model_vram_gb=profile.minimum_gpu_vram_gb,
            )
        except GPUPreflightError as exc:
            rejected.append(f"{profile.profile_id}: {exc}")
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
