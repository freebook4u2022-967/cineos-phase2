import pytest

from cineos.atlas import connected_benchmark_fixture_preflight as preflight
from cineos.atlas.native_request import NativeShotRequest


def _request(
    index: int,
    *,
    lead_reference: str = "lead-ref",
    partner_reference: str = "partner-ref",
    identity_challenge: bool = True,
) -> NativeShotRequest:
    tags = ["identity_consistency"] if identity_challenge else []
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="identity-binding",
        camera={},
        characters=[
            {
                "character_id": "lead",
                "approved_reference_ids": [lead_reference],
            },
            {
                "character_id": "partner",
                "approved_reference_ids": [partner_reference],
            },
        ],
        environment={},
        wardrobe=[],
        props=[],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={},
        approved_reference_ids=[lead_reference, partner_reference],
        deterministic_seed=9000 + index,
        renderer_requirements={},
        metadata={preflight.COMPETITIVE_CHALLENGE_METADATA_KEY: tags},
    )
    request.refresh_hash()
    return request


def test_identity_consistency_evidence_binds_reference_to_same_character():
    requests = [_request(0), _request(1), _request(2, identity_challenge=False)]

    evidence = preflight._validate_identity_consistency_bindings(requests)

    assert evidence["shot-0"] == [
        {"character_id": "lead", "approved_reference_id": "lead-ref"},
        {"character_id": "partner", "approved_reference_id": "partner-ref"},
    ]
    assert evidence["shot-1"] == evidence["shot-0"]


def test_identity_consistency_rejects_reference_transferred_to_other_character():
    first = _request(0)
    second = _request(1)
    second.characters = [
        {"character_id": "lead", "approved_reference_ids": ["partner-ref"]},
        {"character_id": "partner", "approved_reference_ids": ["lead-ref"]},
    ]
    second.refresh_hash()

    with pytest.raises(
        preflight.ConnectedBenchmarkFixturePreflightError,
        match="no character/reference identity binding persists",
    ):
        preflight._validate_identity_consistency_bindings([first, second])


def test_single_character_legacy_shot_level_reference_remains_supported():
    first = _request(0)
    second = _request(1)
    for request in (first, second):
        request.characters = [{"character_id": "lead"}]
        request.approved_reference_ids = ["lead-ref"]
        request.refresh_hash()

    evidence = preflight._validate_identity_consistency_bindings([first, second])

    assert evidence["shot-0"] == [
        {"character_id": "lead", "approved_reference_id": "lead-ref"}
    ]
