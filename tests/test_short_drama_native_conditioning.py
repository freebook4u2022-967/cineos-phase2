import json

from cineos.assets.storage import load as load_asset_registry
from cineos.cinedna import CineDNARegistry
from cineos.short_drama import DramaBrief, ShortDramaOrchestrator
from cineos.short_drama.character_approval import approve_character_files
from cineos.short_drama.integration import write_production_artifacts
from cineos.short_drama.native_conditioning import (
    CharacterConsistencyError,
    build_character_consistency_conditioning,
)


def _identity(path):
    path.write_text(
        json.dumps(
            {
                "display_name": "Protagonist",
                "face": {"invariants": ["same facial identity"]},
                "body": {"build": "locked"},
                "constraints": {
                    "immutable_facial_traits": ["face geometry"],
                    "immutable_body_traits": ["body silhouette"],
                    "forbidden_changes": ["identity drift"],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _project(tmp_path):
    plan = ShortDramaOrchestrator().plan(
        DramaBrief(premise="A mysterious message arrives.", duration_seconds=60)
    )
    return write_production_artifacts(plan, tmp_path)


def test_identity_lock_compiles_to_native_character_conditioning(tmp_path):
    artifacts = _project(tmp_path)
    identity = _identity(tmp_path / "identity.json")
    profiles_path = tmp_path / "cinedna.json"
    approve_character_files(
        artifacts["asset_registry"],
        "Protagonist",
        "refs/full.png",
        identity,
        profiles_path=profiles_path,
        view_type="full-body",
    )
    result = approve_character_files(
        artifacts["asset_registry"],
        "Protagonist",
        "refs/front.png",
        identity,
        profiles_path=profiles_path,
        view_type="front",
    )

    assets = load_asset_registry(artifacts["asset_registry"])
    character = assets.retrieve(result["character_id"])
    profile = CineDNARegistry.load(profiles_path).retrieve(result["character_id"])
    conditioning = build_character_consistency_conditioning(character, profile)

    assert conditioning.approved_reference_ids[0] == result["primary_reference_id"]
    assert "same facial identity" in conditioning.identity_invariants
    assert "face geometry" in conditioning.identity_invariants
    assert "body silhouette" in conditioning.identity_invariants
    assert "forbid:identity drift" in conditioning.identity_invariants
    assert conditioning.scene_specific_overrides["reference_strategy"] == (
        "ranked-multi-reference"
    )


def test_native_conditioning_rejects_character_without_identity_lock(tmp_path):
    artifacts = _project(tmp_path)
    assets = load_asset_registry(artifacts["asset_registry"])
    character = next(iter(assets.list(kind="character")))

    class Profile:
        character_uuid = character.asset_id
        profile_version = "1.0"
        approved_reference_ids = []

    try:
        build_character_consistency_conditioning(character, Profile())
    except CharacterConsistencyError as error:
        assert "identity lock" in str(error)
    else:
        raise AssertionError("native conditioning accepted an unlocked character")
