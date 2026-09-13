import pytest

from cineos.atlas.foundation_profiles import (
    EXTERNAL_PRETRAINED_FOUNDATION,
    WAN22_I2V_A14B_PROFILE,
    WAN22_TI2V_5B_PROFILE,
)
from cineos.atlas.foundation_selection import (
    FoundationSelectionError,
    select_foundation_profile,
)


def test_quality_selection_prefers_a14b_for_i2v_when_80gb_is_available():
    selected = select_foundation_profile(
        required_features={"image_to_video"},
        available_vram_gb=80.0,
    )

    assert selected is WAN22_I2V_A14B_PROFILE
    assert selected.origin == EXTERNAL_PRETRAINED_FOUNDATION


def test_quality_selection_falls_back_to_5b_i2v_when_only_24gb_is_available():
    selected = select_foundation_profile(
        required_features={"image_to_video"},
        available_vram_gb=24.0,
    )

    assert selected is WAN22_TI2V_5B_PROFILE


def test_text_to_video_never_selects_i2v_only_a14b_profile():
    selected = select_foundation_profile(
        required_features={"text_to_video"},
        available_vram_gb=96.0,
    )

    assert selected is WAN22_TI2V_5B_PROFILE


def test_low_memory_selection_can_prefer_lower_vram_profile_explicitly():
    selected = select_foundation_profile(
        required_features={"image_to_video"},
        available_vram_gb=96.0,
        prefer_quality=False,
    )

    assert selected is WAN22_TI2V_5B_PROFILE


def test_selection_fails_closed_when_no_profile_fits_available_vram():
    with pytest.raises(FoundationSelectionError, match="no pinned foundation profile"):
        select_foundation_profile(
            required_features={"image_to_video"},
            available_vram_gb=23.99,
        )


def test_selection_rejects_unknown_feature_requirements():
    with pytest.raises(
        FoundationSelectionError,
        match="unsupported foundation feature requirement",
    ):
        select_foundation_profile(
            required_features={"video_to_video"},
            available_vram_gb=96.0,
        )


def test_selection_rejects_empty_feature_requirements():
    with pytest.raises(
        FoundationSelectionError,
        match="at least one required foundation feature",
    ):
        select_foundation_profile(
            required_features=set(),
            available_vram_gb=96.0,
        )


def test_selection_rejects_nonpositive_or_boolean_vram():
    for value in (0.0, -1.0, False):
        with pytest.raises(FoundationSelectionError, match="available_vram_gb"):
            select_foundation_profile(
                required_features={"image_to_video"},
                available_vram_gb=value,
            )
