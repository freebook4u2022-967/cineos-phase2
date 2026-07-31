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
from cineos.cli.main import main


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
