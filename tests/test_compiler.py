import json

import pytest

from cineos.compiler import (
    PackageValidationError,
    compile,
    load,
    save,
    serialize,
    verify,
)
from cineos.core import (
    Character,
    Environment,
    MovieProject,
    Prop,
    Scene,
    Shot,
    Timeline,
)


def make_project() -> MovieProject:
    character = Character("character-ada", "Ada", metadata={"costume": "blue"})
    location = Environment("location-stage", "Stage")
    prop = Prop("prop-key", "Key")
    shot = Shot(
        "shot-1",
        camera="A",
        action="Ada finds the key.",
        duration=2.0,
        references=[character.asset_id, prop.asset_id],
    )
    scene = Scene(
        "scene-1",
        "Discovery",
        shots=[shot],
        location=location.asset_id,
        characters=[character.asset_id],
        duration=2.0,
    )
    timeline = Timeline([scene.scene_id], {scene.scene_id: [shot.shot_id]})
    return MovieProject(
        "Film",
        "Director",
        characters=[character],
        locations=[location],
        props=[prop],
        scenes=[scene],
        timeline=timeline,
    )


def test_compile_contains_all_manifests_and_deterministic_hashes() -> None:
    first = compile(make_project())
    second = compile(make_project())

    assert first == second
    assert first.project_metadata["title"] == "Film"
    assert first.scene_manifest[0]["shots"] == ["shot-1"]
    assert first.shot_manifest[0]["scene_id"] == "scene-1"
    assert first.character_manifest[0]["asset_id"] == "character-ada"
    assert first.location_manifest[0]["asset_id"] == "location-stage"
    assert {asset["type"] for asset in first.asset_manifest} == {
        "character",
        "location",
        "prop",
    }
    assert len(first.content_hashes["package"]) == 64
    assert verify(first)


def test_save_and_load_canonical_json(tmp_path) -> None:
    package = compile(make_project())
    destination = tmp_path / "film-package.json"

    data = save(package, destination)
    loaded = load(destination)

    assert loaded == package
    assert data == serialize(package)
    assert destination.read_text(encoding="utf-8") == data + "\n"
    assert load(data) == package
    assert load(json.loads(data)) == package


def test_verify_detects_modified_content() -> None:
    package = compile(make_project())
    package.shot_manifest[0]["action"] = "Tampered"

    with pytest.raises(PackageValidationError, match="content hashes"):
        verify(package)


def test_compile_rejects_invalid_project() -> None:
    project = make_project()
    project.timeline.scene_order.clear()

    with pytest.raises(ValueError, match="timeline scene order"):
        compile(project)
