from types import SimpleNamespace

import pytest

from cineos.atlas.diffusers_video import DiffusersVideoError
from cineos.atlas.production_diffusers import ProductionDiffusersVideoRenderer


def _request(*, approved_reference_ids, characters):
    return SimpleNamespace(
        approved_reference_ids=tuple(approved_reference_ids),
        characters=tuple(characters),
    )


def test_multi_character_reference_lineage_rejects_unowned_global_reference():
    request = _request(
        approved_reference_ids=("hero-face", "partner-face", "mystery-face"),
        characters=(
            {
                "character_uuid": "hero",
                "approved_reference_ids": ["hero-face"],
            },
            {
                "character_uuid": "partner",
                "approved_reference_ids": ["partner-face"],
            },
        ),
    )

    with pytest.raises(DiffusersVideoError, match="unowned: mystery-face"):
        ProductionDiffusersVideoRenderer._validate_character_reference_lineage(request)


def test_multi_character_reference_lineage_accepts_complete_unique_ownership():
    request = _request(
        approved_reference_ids=("hero-face", "hero-profile", "partner-face"),
        characters=(
            {
                "character_uuid": "hero",
                "approved_reference_ids": ["hero-face", "hero-profile"],
            },
            {
                "character_uuid": "partner",
                "approved_reference_ids": ["partner-face"],
            },
        ),
    )

    ProductionDiffusersVideoRenderer._validate_character_reference_lineage(request)


def test_multi_character_reference_lineage_rejects_cross_character_shared_reference():
    request = _request(
        approved_reference_ids=("shared-face",),
        characters=(
            {
                "character_uuid": "hero",
                "approved_reference_ids": ["shared-face"],
            },
            {
                "character_uuid": "partner",
                "approved_reference_ids": ["shared-face"],
            },
        ),
    )

    with pytest.raises(DiffusersVideoError, match="ambiguously assigned"):
        ProductionDiffusersVideoRenderer._validate_character_reference_lineage(request)


def test_single_character_reference_lineage_rejects_unowned_global_reference():
    request = _request(
        approved_reference_ids=("hero-face", "hero-profile"),
        characters=(
            {
                "character_uuid": "hero",
                "approved_reference_ids": ["hero-face"],
            },
        ),
    )

    with pytest.raises(
        DiffusersVideoError,
        match="single-character.*unowned: hero-profile",
    ):
        ProductionDiffusersVideoRenderer._validate_character_reference_lineage(request)


def test_single_character_reference_lineage_accepts_complete_ownership():
    request = _request(
        approved_reference_ids=("hero-face", "hero-profile"),
        characters=(
            {
                "character_uuid": "hero",
                "approved_reference_ids": ["hero-face", "hero-profile"],
            },
        ),
    )

    ProductionDiffusersVideoRenderer._validate_character_reference_lineage(request)
