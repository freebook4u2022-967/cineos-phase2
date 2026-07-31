"""Canonical asset integrity and CLI coverage."""

import hashlib
import json

import pytest

from cineos.assets import (
    ApprovalStatus,
    AssetRegistry,
    Character,
    Environment,
    Prop,
    ReferenceImage,
    Wardrobe,
)
from cineos.assets.exceptions import DuplicateAssetError
from cineos.assets.storage import load, save
from cineos.assets.validator import AssetValidator
from cineos.cli.commands import load_project
from cineos.cli.main import main
from cineos.compiler import compile as compile_project


def test_reference_round_trip_and_checksum_validation(tmp_path):
    media = tmp_path / "hero.png"
    media.write_bytes(b"not copied")
    hero = Character(name="Hero")
    hero.add_reference(
        str(media),
        view_type="front",
        checksum=hashlib.sha256(media.read_bytes()).hexdigest(),
        dimensions=(1920, 1080),
        approval_status=ApprovalStatus.APPROVED,
        source="still",
    )
    registry = AssetRegistry()
    registry.register(hero)
    first = save(registry, tmp_path / "one.json")
    second = save(load(first), tmp_path / "two.json")
    assert first.read_bytes() == second.read_bytes()
    assert AssetValidator().validate(load(first), check_files=True) == []
    media.write_bytes(b"changed")
    assert "checksum mismatch" in " ".join(
        AssetValidator().validate(load(first), check_files=True)
    )


def test_duplicate_names_and_invalid_relationships():
    registry = AssetRegistry()
    hero = registry.register(Character(name="Hero"))
    with pytest.raises(DuplicateAssetError):
        registry.register(Character(name="hero"))
    coat = registry.register(Wardrobe(name="Coat"))
    registry.relate(coat, hero, "character-wardrobe")
    assert "invalid character-wardrobe" in " ".join(registry.validate())


def test_missing_reference_is_reported_by_filesystem_check(tmp_path):
    registry = AssetRegistry()
    registry.register(
        Prop(
            name="Key",
            reference_images=[
                ReferenceImage(
                    file_path=str(tmp_path / "absent.png"), checksum="0" * 64
                )
            ],
        )
    )
    assert registry.validate() == []
    assert "missing reference file" in " ".join(
        AssetValidator().validate(registry, check_files=True)
    )


def test_cli_add_show_and_json(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "environment.json"
    manifest.write_text(json.dumps({"name": "Stage", "tags": ["interior"]}))
    assert main(["assets", "add-environment", str(manifest), "--json"]) == 0
    asset_id = json.loads(capsys.readouterr().out)["asset"]["asset_id"]
    assert main(["assets", "show", asset_id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["asset"]["name"] == "Stage"
    assert isinstance(load(tmp_path / "assets.json").get(asset_id), Environment)


def test_project_loads_external_assets_and_package_uses_only_ids(tmp_path):
    registry = AssetRegistry()
    hero = registry.register(Character(name="Hero"))
    hero.add_reference("media/hero-front.png", checksum="0" * 64)
    save(registry, tmp_path / "assets.json")
    project_path = tmp_path / "project.json"
    project_path.write_text(
        json.dumps(
            {
                "title": "Film",
                "author": "Maker",
                "asset_registry": "assets.json",
                "asset_ids": [str(hero.asset_id)],
            }
        )
    )

    project = load_project(project_path)
    package = compile_project(project)

    assert project.asset_registry.retrieve(hero.asset_id).name == "Hero"
    assert package.asset_manifest == [
        {
            "asset_id": str(hero.asset_id),
            "type": "character",
            "name": "Hero",
            "version": 1,
            "content_hash": hero.content_hash,
        }
    ]
    assert "hero-front.png" not in json.dumps(package.asset_manifest)


def test_duplicate_reference_uuids_are_rejected_by_validation():
    shared = ReferenceImage(file_path="hero.png")
    registry = AssetRegistry()
    registry.register(Character(name="Hero", reference_images=[shared]))
    registry.register(Environment(name="Stage", reference_images=[shared.copy()]))

    assert f"duplicate reference UUID: {shared.reference_id}" in registry.validate()
