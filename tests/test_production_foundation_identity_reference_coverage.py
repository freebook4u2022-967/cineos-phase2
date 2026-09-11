"""Regression tests for competitive identity-reference coverage at GPU selection."""

from types import SimpleNamespace

import pytest

from cineos.atlas.foundation_profiles import WAN22_I2V_A14B_PROFILE
from cineos.atlas.gpu_preflight import GPUDeviceProfile
from cineos.atlas.production_foundation_selection import (
    ProductionFoundationSelectionError,
    select_strongest_production_foundation,
)


def _gpu() -> GPUDeviceProfile:
    return GPUDeviceProfile(
        index=0,
        name="test-gpu",
        compute_capability=(9, 0),
        total_vram_gb=100.0,
        free_vram_gb=96.0,
        supports_bfloat16=True,
    )


def _request(*refs: str, tagged: bool = True):
    return SimpleNamespace(
        approved_reference_ids=tuple(refs),
        camera={"resolution": (832, 480), "fps": 16.0, "duration": 2.0},
        characters=(
            {"character_id": "hero"},
            {"character_id": "partner"},
        ),
        metadata=(
            {
                "competitive_challenges": [
                    "identity_consistency",
                    "multi_character_interaction",
                ]
            }
            if tagged
            else {}
        ),
    )


def test_competitive_multi_character_shot_rejects_single_identity_reference():
    with pytest.raises(
        ProductionFoundationSelectionError,
        match=(
            "requires at least one distinct approved reference per conditioned "
            "character: characters=2 references=1"
        ),
    ):
        select_strongest_production_foundation((_gpu(),), (_request("hero"),))


def test_competitive_multi_character_shot_accepts_distinct_reference_coverage():
    selection = select_strongest_production_foundation(
        (_gpu(),),
        (_request("hero", "partner"),),
    )

    assert selection.profile is WAN22_I2V_A14B_PROFILE
    assert selection.fallback_used is False


def test_competitive_multi_character_shot_rejects_duplicate_reference_ids():
    with pytest.raises(
        ProductionFoundationSelectionError,
        match="contains duplicate approved reference identities",
    ):
        select_strongest_production_foundation(
            (_gpu(),),
            (_request("hero", "hero"),),
        )


def test_untagged_legacy_request_preserves_historical_single_reference_behavior():
    selection = select_strongest_production_foundation(
        (_gpu(),),
        (_request("hero", tagged=False),),
    )

    assert selection.profile is WAN22_I2V_A14B_PROFILE
