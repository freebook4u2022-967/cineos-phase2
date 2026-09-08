"""Regression coverage for quality-first production foundation selection."""

from types import SimpleNamespace

import pytest

from cineos.atlas.foundation_profiles import (
    WAN22_I2V_A14B_PROFILE,
    WAN22_TI2V_5B_PROFILE,
)
from cineos.atlas.gpu_preflight import GPUDeviceProfile
from cineos.atlas.production_foundation_selection import (
    ProductionFoundationSelectionError,
    select_strongest_production_foundation,
)


def _gpu(total: float, free: float | None = None) -> GPUDeviceProfile:
    return GPUDeviceProfile(
        index=0,
        name="test-gpu",
        compute_capability=(9, 0),
        total_vram_gb=total,
        free_vram_gb=total if free is None else free,
        supports_bfloat16=True,
    )


def _request(*refs: str):
    return SimpleNamespace(approved_reference_ids=tuple(refs))


def test_prefers_a14b_when_every_shot_is_image_conditioned_and_vram_floor_is_met():
    selection = select_strongest_production_foundation(
        (_gpu(96.0, 90.0),),
        (_request("hero"), _request("hero", "partner")),
    )

    assert selection.profile is WAN22_I2V_A14B_PROFILE
    assert selection.fallback_used is False
    assert selection.rejected_profiles == ()
    assert selection.to_dict()["origin"] == "external_pretrained_foundation"


def test_falls_back_to_5b_below_unvalidated_a14b_vram_floor():
    selection = select_strongest_production_foundation(
        (_gpu(48.0, 44.0),),
        (_request("hero"), _request("hero")),
    )

    assert selection.profile is WAN22_TI2V_5B_PROFILE
    assert selection.fallback_used is True
    assert any("production floor" in reason for reason in selection.rejected_profiles)


def test_falls_back_to_5b_when_any_shot_lacks_image_conditioning():
    selection = select_strongest_production_foundation(
        (_gpu(96.0, 90.0),),
        (_request("hero"), _request()),
    )

    assert selection.profile is WAN22_TI2V_5B_PROFILE
    assert selection.fallback_used is True
    assert any("image conditioning" in reason for reason in selection.rejected_profiles)


def test_fails_closed_when_no_approved_profile_can_fit():
    with pytest.raises(
        ProductionFoundationSelectionError,
        match="no approved pinned production video foundation can safely execute",
    ):
        select_strongest_production_foundation(
            (_gpu(4.0, 3.0),),
            (_request("hero"), _request("hero")),
        )
