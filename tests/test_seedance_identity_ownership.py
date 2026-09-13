from __future__ import annotations

import pytest

from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.seedance_style_challenge import (
    SeedanceStyleChallengeError,
    _validate_sequence_identity_grounding,
)


def _request(
    shot_id: str,
    *,
    characters: list[dict[str, object]],
    approved_reference_ids: list[str],
) -> NativeShotRequest:
    return NativeShotRequest(
        shot_id=shot_id,
        scene_id="scene-identity",
        camera={},
        characters=characters,
        environment=None,
        wardrobe=[],
        props=[],
        continuity={},
        performance={},
        approved_reference_ids=approved_reference_ids,
        deterministic_seed=1,
        renderer_requirements={},
    )


def test_identity_grounding_accepts_persistent_character_reference_owner() -> None:
    requests = [
        _request(
            "shot-1",
            characters=[
                {
                    "character_uuid": "hero",
                    "approved_reference_ids": ["hero-ref"],
                }
            ],
            approved_reference_ids=["hero-ref"],
        ),
        _request(
            "shot-2",
            characters=[
                {
                    "character_uuid": "hero",
                    "approved_reference_ids": ["hero-ref"],
                }
            ],
            approved_reference_ids=["hero-ref"],
        ),
    ]

    _validate_sequence_identity_grounding(
        requests,
        [("identity_consistency",), ("identity_consistency",)],
    )


def test_identity_grounding_rejects_reference_owner_swap() -> None:
    requests = [
        _request(
            "shot-1",
            characters=[
                {"character_uuid": "hero", "approved_reference_ids": ["ref-a"]},
                {"character_uuid": "partner", "approved_reference_ids": ["ref-b"]},
            ],
            approved_reference_ids=["ref-a", "ref-b"],
        ),
        _request(
            "shot-2",
            characters=[
                {"character_uuid": "hero", "approved_reference_ids": ["ref-b"]},
                {"character_uuid": "partner", "approved_reference_ids": ["ref-a"]},
            ],
            approved_reference_ids=["ref-a", "ref-b"],
        ),
    ]

    with pytest.raises(
        SeedanceStyleChallengeError,
        match="character-to-reference ownership pairs persists",
    ):
        _validate_sequence_identity_grounding(
            requests,
            [("identity_consistency",), ("identity_consistency",)],
        )


def test_identity_grounding_preserves_legacy_global_reference_compatibility() -> None:
    requests = [
        _request(
            "shot-1",
            characters=[{"character_uuid": "hero"}],
            approved_reference_ids=["hero-ref"],
        ),
        _request(
            "shot-2",
            characters=[{"character_uuid": "hero"}],
            approved_reference_ids=["hero-ref"],
        ),
    ]

    _validate_sequence_identity_grounding(
        requests,
        [("identity_consistency",), ("identity_consistency",)],
    )


def test_identity_grounding_fails_closed_when_local_ownership_does_not_persist() -> (
    None
):
    requests = [
        _request(
            "shot-1",
            characters=[
                {"character_uuid": "hero", "approved_reference_ids": ["hero-a"]}
            ],
            approved_reference_ids=["hero-a", "global-ref"],
        ),
        _request(
            "shot-2",
            characters=[
                {"character_uuid": "hero", "approved_reference_ids": ["hero-b"]}
            ],
            approved_reference_ids=["hero-b", "global-ref"],
        ),
    ]

    with pytest.raises(
        SeedanceStyleChallengeError,
        match="character-to-reference ownership pairs persists",
    ):
        _validate_sequence_identity_grounding(
            requests,
            [("identity_consistency",), ("identity_consistency",)],
        )
