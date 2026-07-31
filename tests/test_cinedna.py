"""CineDNA identity, persistence, integration, and CLI coverage."""

import json
from dataclasses import replace
from uuid import UUID

import pytest

from cineos.assets import AssetRegistry, Character, ReferenceImage
from cineos.assets.storage import save as save_assets
from cineos.cinedna import (
    CineDNABuilder,
    CineDNARegistry,
    ConflictingIdentityDataError,
    MissingIdentityDataError,
    deserialize,
    serialize,
)
from cineos.cli.main import main
from cineos.compiler import compile
from cineos.compiler import deserialize as deserialize_package
from cineos.compiler import serialize as serialize_package
from cineos.core import MovieProject

CHARACTER_ID = UUID("11111111-1111-4111-8111-111111111111")
REFERENCE_ID = UUID("22222222-2222-4222-8222-222222222222")


def character(*, approved: bool = True) -> Character:
    return Character(
        asset_id=CHARACTER_ID,
        name="Alex",
        reference_images=[
            ReferenceImage(
                "alex.png",
                reference_id=REFERENCE_ID,
                approval_status="approved" if approved else "pending",
            )
        ],
        metadata={
            "cinedna": {
                "face": {
                    "facial_feature_descriptors": {"jaw": "angular"},
                    "invariants": ["jaw shape"],
                },
                "body": {"height": "175 cm", "dominant_hand": "left"},
                "expressions": {"happy": {"description": "broad smile"}},
            }
        },
    )


def test_build_is_deterministic_and_uses_only_approved_references():
    first = CineDNABuilder().build(character())
    second = CineDNABuilder().build(character())
    assert serialize(first) == serialize(second)
    assert first.approved_reference_ids == [str(REFERENCE_ID)]


def test_builder_rejects_unapproved_and_missing_identity_data():
    with pytest.raises(MissingIdentityDataError):
        CineDNABuilder().build(character(approved=False))
    asset = character()
    asset.metadata = {}
    with pytest.raises(MissingIdentityDataError):
        CineDNABuilder().build(asset)


def test_builder_rejects_conflicting_wardrobe_locks():
    asset = character()
    asset.metadata["cinedna"]["wardrobe"] = [
        {
            "wardrobe_asset_id": "day",
            "scene_applicability": ["scene-1"],
            "continuity_lock": True,
        },
        {
            "wardrobe_asset_id": "night",
            "scene_applicability": ["scene-1"],
            "continuity_lock": True,
        },
    ]
    with pytest.raises(ConflictingIdentityDataError):
        CineDNABuilder().build(asset)


def test_serialization_round_trip_and_hash_verification():
    profile = CineDNABuilder().build(character())
    loaded = deserialize(serialize(profile))
    assert loaded == profile
    value = json.loads(serialize(profile))
    value["display_name"] = "Tampered"
    with pytest.raises(ValueError, match="content hash"):
        deserialize(json.dumps(value))


def test_registry_versions_save_and_resolve(tmp_path):
    profile = CineDNABuilder().build(character())
    registry = CineDNARegistry()
    registry.register(profile)
    version_two = replace(profile, profile_version="2.0", metadata={"note": "v2"})
    version_two.refresh_content_hash()
    registry.update(version_two)
    path = tmp_path / "profiles.json"
    registry.save(path)
    restored = CineDNARegistry.load(path)
    assert restored.retrieve(CHARACTER_ID).profile_version == "2.0"
    assert len(restored.version(CHARACTER_ID)) == 2
    assert restored.validate() == []


def test_project_and_film_package_preserve_cinedna_ids():
    project = MovieProject("Movie", "Author", cinedna_ids=[CHARACTER_ID])
    package = compile(project)
    assert package.cinedna_ids == [str(CHARACTER_ID)]
    assert deserialize_package(serialize_package(package)).cinedna_ids == [
        str(CHARACTER_ID)
    ]


def test_cli_build_list_show_validate_and_export(tmp_path, capsys):
    assets = AssetRegistry()
    assets.register(character())
    assets_path = tmp_path / "assets.json"
    profiles_path = tmp_path / "profiles.json"
    exported_path = tmp_path / "profile.json"
    save_assets(assets, assets_path)
    common = [
        "cinedna",
        "--registry",
        str(assets_path),
        "--profiles",
        str(profiles_path),
    ]
    assert main([*common, "build", str(CHARACTER_ID), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert main([*common, "list"]) == 0
    assert str(CHARACTER_ID) in capsys.readouterr().out
    assert main([*common, "show", str(CHARACTER_ID)]) == 0
    assert "Alex" in capsys.readouterr().out
    assert main([*common, "validate", str(CHARACTER_ID)]) == 0
    capsys.readouterr()
    assert (
        main([*common, "export", str(CHARACTER_ID), "--output", str(exported_path)])
        == 0
    )
    assert deserialize(exported_path.read_text()).character_uuid == CHARACTER_ID
