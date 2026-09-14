from __future__ import annotations

import pytest

from cineos.atlas.native_request import NativeShotRequest


def _request(
    *,
    approved_reference_ids: list[str],
    characters: list[dict] | None = None,
) -> NativeShotRequest:
    return NativeShotRequest(
        shot_id="shot-1",
        scene_id="scene-1",
        camera={},
        characters=characters or [],
        environment=None,
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=approved_reference_ids,
        deterministic_seed=7,
        renderer_requirements={},
    )


def test_shot_level_duplicates_remain_renderer_validated() -> None:
    request = _request(approved_reference_ids=["hero-ref", "hero-ref"])

    content_hash = request.refresh_hash()

    assert len(content_hash) == 64
    assert request.content_hash_is_current()


def test_native_request_rejects_duplicate_character_reference_ids() -> None:
    request = _request(
        approved_reference_ids=["hero-ref"],
        characters=[
            {
                "character_uuid": "hero",
                "approved_reference_ids": ["hero-ref", "hero-ref"],
            }
        ],
    )

    with pytest.raises(ValueError, match="duplicate reference IDs"):
        request.refresh_hash()


def test_native_request_rejects_character_reference_not_approved_for_shot() -> None:
    request = _request(
        approved_reference_ids=["hero-ref"],
        characters=[
            {
                "character_uuid": "hero",
                "approved_reference_ids": ["unapproved-ref"],
            }
        ],
    )

    with pytest.raises(ValueError, match="unapproved shot reference 'unapproved-ref'"):
        request.refresh_hash()


def test_native_request_rejects_reference_shared_across_characters() -> None:
    request = _request(
        approved_reference_ids=["shared-ref"],
        characters=[
            {"character_uuid": "hero", "approved_reference_ids": ["shared-ref"]},
            {
                "character_uuid": "partner",
                "approved_reference_ids": ["shared-ref"],
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "approved reference 'shared-ref' is assigned to multiple characters: "
            r"characters\[0\] and characters\[1\]"
        ),
    ):
        request.refresh_hash()


def test_native_request_accepts_unique_multi_character_reference_ids() -> None:
    request = _request(
        approved_reference_ids=["hero-ref", "partner-ref"],
        characters=[
            {"character_uuid": "hero", "approved_reference_ids": ["hero-ref"]},
            {
                "character_uuid": "partner",
                "approved_reference_ids": ["partner-ref"],
            },
        ],
    )

    content_hash = request.refresh_hash()

    assert len(content_hash) == 64
    assert request.content_hash_is_current()
