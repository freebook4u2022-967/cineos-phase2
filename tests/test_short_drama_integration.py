from cineos.compiler import verify
from cineos.short_drama import (
    DramaBrief,
    ShortDramaOrchestrator,
    compile_drama_plan,
    write_production_artifacts,
)


def _plan():
    brief = DramaBrief(
        premise="A man receives a message from his wife who died three years ago.",
        duration_seconds=180,
        genre="mystery",
        tone="tense and intimate",
    )
    return ShortDramaOrchestrator().plan(brief)


def test_drama_plan_compiles_through_existing_film_compiler():
    project, package = compile_drama_plan(_plan())

    verify(package)
    assert project.timeline.scene_order == [
        scene.scene_id for scene in project.scenes
    ]
    assert package.project_metadata["title"].startswith("A man receives")
    assert len(package.scene_manifest) == 5
    assert len(package.shot_manifest) == 5
    assert len(package.asset_manifest) >= len(project.characters)
    assert package.cinedna_ids == []


def test_character_assets_are_cinedna_ready_but_not_fabricated():
    project, _ = compile_drama_plan(_plan())
    canonical_characters = project.asset_registry.list(kind="character")

    assert canonical_characters
    assert project.cinedna_ids == []
    for character in canonical_characters:
        identity = character.metadata["cinedna"]
        assert identity["status"] == "pending-approved-reference"
        assert identity["required_identity_fields"] == ["face", "body"]
        assert character.references == []


def test_production_artifacts_are_written(tmp_path):
    paths = write_production_artifacts(_plan(), tmp_path)

    assert paths["drama_package"].is_file()
    assert paths["asset_registry"].is_file()
    assert paths["film_package"].is_file()
