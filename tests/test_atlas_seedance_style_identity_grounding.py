import pytest

from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.seedance_style_challenge import (
    CHALLENGE_METADATA_KEY,
    REQUIRED_CHALLENGES,
    SeedanceStyleChallengeError,
    validate_challenge_coverage,
)


def _request(index: int, *, persistent_characters: bool) -> NativeShotRequest:
    if persistent_characters:
        characters = [{"character_id": "lead"}, {"character_id": "partner"}]
    else:
        characters = [
            {"character_id": f"lead-{index}"},
            {"character_id": f"partner-{index}"},
        ]
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-seedance-identity",
        camera={"movement": "whip_pan"},
        characters=characters,
        environment={"location": "street"},
        wardrobe=[],
        props=[{"prop_id": "case"}],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={},
        approved_reference_ids=["lead-ref", "partner-ref"],
        deterministic_seed=9100 + index,
        renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
        metadata={CHALLENGE_METADATA_KEY: list(REQUIRED_CHALLENGES)},
    )
    request.refresh_hash()
    return request


def test_identity_challenge_requires_persistent_conditioned_character_identity():
    requests = [_request(index, persistent_characters=False) for index in range(5)]

    with pytest.raises(
        SeedanceStyleChallengeError,
        match="conditioned character identities persists into another connected shot",
    ):
        validate_challenge_coverage(requests)


def test_identity_challenge_accepts_persistent_character_and_reference():
    requests = [_request(index, persistent_characters=True) for index in range(5)]

    coverage = validate_challenge_coverage(requests)

    assert coverage.complete is True
    assert (
        coverage.to_dict()["schema"] == "cineos-seedance-style-challenge-coverage/0.2"
    )
