import json
from dataclasses import dataclass

import pytest

from cineos.compiler import (
    FilmPackage,
    PackageValidationError,
    compile,
    load,
    save,
    verify,
)
from cineos.compiler.serializer import dumps, loads


@dataclass
class MovieProject:
    metadata: dict[str, object]
    scenes: list[dict[str, object]]
    shots: list[dict[str, object]]
    characters: list[dict[str, object]]
    locations: list[dict[str, object]]
    assets: list[dict[str, object]]
    timeline: list[dict[str, object]]


def project() -> MovieProject:
    return MovieProject(
        metadata={"title": "Example", "fps": 24},
        scenes=[{"id": "scene-b"}, {"id": "scene-a"}],
        shots=[{"id": "shot-a", "scene_id": "scene-a"}],
        characters=[{"id": "hero", "name": "Ada"}],
        locations=[{"id": "lab"}],
        assets=[{"id": "plate", "uri": "assets/plate.exr"}],
        timeline=[{"shot_id": "shot-a", "start_frame": 0, "end_frame": 23}],
    )


def test_compile_produces_all_manifests_and_valid_hashes() -> None:
    package = compile(project())

    assert isinstance(package, FilmPackage)
    assert package.version == "1.0"
    assert package.project_metadata["title"] == "Example"
    assert [scene["id"] for scene in package.scene_manifest] == [
        "scene-a",
        "scene-b",
    ]
    assert package.shot_manifest and package.character_manifest
    assert package.location_manifest and package.asset_manifest
    assert package.timeline_manifest
    assert verify(package)
    assert set(package.content_hashes) == {
        "project_metadata",
        "scene_manifest",
        "shot_manifest",
        "character_manifest",
        "location_manifest",
        "asset_manifest",
        "timeline_manifest",
        "package",
    }


def test_compile_is_deterministic_across_mapping_and_input_order() -> None:
    first = compile(project())
    reordered = project()
    reordered.scenes.reverse()
    second = compile(reordered)

    assert dumps(first) == dumps(second)
    assert first.content_hashes == second.content_hashes


def test_save_and_load_round_trip(tmp_path) -> None:
    package = compile(project())
    path = save(package, tmp_path / "movie.film.json")

    assert path.read_text(encoding="utf-8").endswith("\n")
    assert load(path) == package
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == "1.0"


def test_load_rejects_tampered_package(tmp_path) -> None:
    path = save(compile(project()), tmp_path / "movie.film.json")
    value = json.loads(path.read_text(encoding="utf-8"))
    value["project_metadata"]["title"] = "Tampered"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(PackageValidationError, match="hash mismatch"):
        load(path)


def test_serialization_rejects_invalid_document() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        loads("[]")


def test_verify_returns_false_for_unsupported_version() -> None:
    package = compile(project())
    unsupported = FilmPackage(
        **{
            **package.to_dict(),
            "version": "99.0",
            "content_hashes": package.content_hashes,
        }
    )

    assert not verify(unsupported)
