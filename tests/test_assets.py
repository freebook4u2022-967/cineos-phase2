"""Tests for asset identity, references, versions, relationships, and storage."""

import json

import pytest

from cineos.assets import AssetRegistry, Character, Prop, Wardrobe
from cineos.assets.storage import load, save
from cineos.cli.main import main
from cineos.core import MovieProject, ProjectValidator


def _registry() -> AssetRegistry:
    registry = AssetRegistry()
    hero = registry.register(Character(name="Hero", tags={"principal"}))
    coat = registry.register(Wardrobe(name="Blue coat"))
    prop = registry.register(Prop(name="Map", metadata={"era": "1920s"}))
    hero.add_reference("references/hero-front.png", label="front")
    hero.add_reference("https://example.test/hero-profile.png", label="profile")
    hero.create_version("approved concept")
    registry.relate(hero, coat, "wears")
    registry.relate(hero, prop, "uses")
    return registry


def test_assets_receive_unique_uuids_and_support_revisions() -> None:
    registry = _registry()
    hero = next(asset for asset in registry.list() if asset.name == "Hero")
    assert len({asset.asset_id for asset in registry.list()}) == 3
    assert len(hero.reference_images) == 2
    assert hero.versions[0].version == 1
    assert hero.versions[0].reference_images is not hero.reference_images
    assert registry.validate() == []


def test_registry_relationships_require_registered_endpoints() -> None:
    registry = AssetRegistry()
    hero = registry.register(Character(name="Hero"))
    with pytest.raises(ValueError, match="endpoints"):
        registry.relate(hero, Prop(name="Key"), "uses")


def test_storage_round_trip_preserves_assets_and_relationships(tmp_path) -> None:
    destination = save(_registry(), tmp_path / "assets.json")
    restored = load(destination)
    assert [asset.name for asset in restored.list()] == ["Hero", "Map", "Blue coat"]
    assert [item.relationship for item in restored.relationships] == ["uses", "wears"]
    assert save(restored, tmp_path / "copy.json").read_text() == destination.read_text()


def test_movie_project_validates_integrated_registry() -> None:
    project = MovieProject("Film", "Maker", asset_registry=_registry())
    assert ProjectValidator().validate(project) == []


def test_assets_cli_lists_validates_and_exports(tmp_path, capsys) -> None:
    source = save(_registry(), tmp_path / "assets.json")
    assert main(["assets", "list", str(source), "--json"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert any(asset["name"] == "Hero" for asset in listing["assets"])
    assert main(["assets", "validate", str(source)]) == 0
    capsys.readouterr()
    destination = tmp_path / "export.json"
    assert main(["assets", "export", str(source), "--output", str(destination)]) == 0
    capsys.readouterr()
    assert json.loads(destination.read_text())["format"] == "cineos-assets-v1"
