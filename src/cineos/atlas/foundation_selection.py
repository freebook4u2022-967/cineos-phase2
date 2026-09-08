"""Deterministic selection of pinned external video foundation profiles.

The selector is intentionally conservative: it only chooses among explicitly pinned,
licensed execution profiles already declared by CINEOS. Selection does not imply that
the underlying weights are CINEOS-native; the returned profile retains its external
pretrained provenance and immutable revision.
"""

from __future__ import annotations

from collections.abc import Iterable

from .foundation_profiles import (
    FoundationExecutionProfile,
    WAN22_I2V_A14B_PROFILE,
    WAN22_TI2V_5B_PROFILE,
)

FOUNDATION_EXECUTION_PROFILES: tuple[FoundationExecutionProfile, ...] = (
    WAN22_TI2V_5B_PROFILE,
    WAN22_I2V_A14B_PROFILE,
)


class FoundationSelectionError(RuntimeError):
    """Raised when no pinned foundation can satisfy an execution requirement."""


def _normalize_required_features(required_features: Iterable[str]) -> frozenset[str]:
    normalized = frozenset(feature.strip() for feature in required_features if feature.strip())
    if not normalized:
        raise FoundationSelectionError("at least one required foundation feature is needed")
    unsupported = normalized - {"text_to_video", "image_to_video"}
    if unsupported:
        raise FoundationSelectionError(
            "unsupported foundation feature requirement: " + ", ".join(sorted(unsupported))
        )
    return normalized


def select_foundation_profile(
    *,
    required_features: Iterable[str],
    available_vram_gb: float,
    prefer_quality: bool = True,
    profiles: Iterable[FoundationExecutionProfile] = FOUNDATION_EXECUTION_PROFILES,
) -> FoundationExecutionProfile:
    """Select the strongest compatible pinned profile that fits observed VRAM.

    Quality preference is deliberately represented by the profile's conservative
    resident-VRAM requirement: among compatible profiles that fit, the higher-capacity
    profile wins. This keeps selection deterministic and auditable without inventing a
    synthetic quality score before real multi-shot benchmark evidence exists.

    When ``prefer_quality`` is false, the lowest-memory compatible profile is selected
    instead. The caller remains responsible for runtime free-memory safety margins and
    actual CUDA execution preflight.
    """

    if isinstance(available_vram_gb, bool) or available_vram_gb <= 0:
        raise FoundationSelectionError("available_vram_gb must be positive")

    required = _normalize_required_features(required_features)
    candidates = [
        profile
        for profile in profiles
        if required.issubset(profile.supported_features)
        and profile.minimum_gpu_vram_gb <= float(available_vram_gb)
    ]
    if not candidates:
        raise FoundationSelectionError(
            "no pinned foundation profile satisfies required features "
            f"{sorted(required)} within {float(available_vram_gb):.2f} GB VRAM"
        )

    candidates.sort(
        key=lambda profile: (
            profile.minimum_gpu_vram_gb,
            max(width * height for width, height in profile.resolutions),
            profile.profile_id,
        ),
        reverse=prefer_quality,
    )
    return candidates[0]


__all__ = [
    "FOUNDATION_EXECUTION_PROFILES",
    "FoundationSelectionError",
    "select_foundation_profile",
]
