import pytest

from cineos.core import (
    AssetRegistry,
    Character,
    Environment,
    MovieProject,
    ProjectValidationError,
    ProjectValidator,
    Scene,
    Shot,
    Timeline,
)


def make_valid_project() -> MovieProject:
    character = Character("character-ada", "Ada")
    location = Environment("environment-stage", "Stage")
    shot = Shot("shot-1", duration=2.0, references=[character.asset_id])
    scene = Scene(
        "scene-1",
        "Opening",
        shots=[shot],
        location=location.asset_id,
        characters=[character.asset_id],
        duration=2.0,
    )
    timeline = Timeline()
    timeline.add_scene(scene.scene_id)
    timeline.add_shot(scene.scene_id, shot.shot_id)
    return MovieProject(
        "Film",
        "Author",
        characters=[character],
        locations=[location],
        scenes=[scene],
        timeline=timeline,
    )


def test_registry_registers_assets_with_unique_ids() -> None:
    registry = AssetRegistry()

    first = registry.register_character("Ada")
    second = registry.register_character("Ada")
    location = registry.register_location("Main Stage")

    assert first.asset_id == "character-ada"
    assert second.asset_id == "character-ada-2"
    assert registry.characters[first.asset_id] is first
    assert registry.environments[location.asset_id] is location


def test_registry_rejects_duplicate_ids_across_asset_types() -> None:
    registry = AssetRegistry()
    registry.register_character(Character("shared", "Ada"))

    with pytest.raises(ValueError, match="duplicate asset ID"):
        registry.register_environment(Environment("shared", "Stage"))


def test_timeline_maintains_order_and_validates_duration() -> None:
    timeline = Timeline()
    timeline.add_scene("scene-1")
    timeline.add_shot("scene-1", "shot-2")
    timeline.add_shot("scene-1", "shot-1", position=0)
    scene = Scene(
        "scene-1",
        "Opening",
        shots=[Shot("shot-1", duration=1), Shot("shot-2", duration=2)],
        duration=4,
    )

    assert timeline.scene_order == ["scene-1"]
    assert timeline.shot_order == {"scene-1": ["shot-1", "shot-2"]}
    assert timeline.validate_durations([scene]) == [
        "scene 'scene-1' duration 4 does not match shot duration 3"
    ]


def test_validator_accepts_consistent_project() -> None:
    project = make_valid_project()

    assert ProjectValidator().validate(project) == []
    assert ProjectValidator().is_valid(project)


def test_validator_reports_ids_references_and_timeline_errors() -> None:
    project = make_valid_project()
    project.scenes.append(
        Scene(
            "scene-1",
            "Duplicate",
            shots=[Shot("shot-1", references=["missing"], duration=1)],
            location="missing",
            characters=["missing"],
            duration=2,
        )
    )

    errors = ProjectValidator().validate(project)

    assert "duplicate scene ID: scene-1" in errors
    assert "duplicate shot ID: shot-1" in errors
    assert any("unknown location" in error for error in errors)
    assert any("unknown character" in error for error in errors)
    assert any("unknown asset" in error for error in errors)
    assert any("duration" in error for error in errors)
    assert "timeline scene order does not match project scenes" in errors
    with pytest.raises(ProjectValidationError):
        ProjectValidator().raise_for_errors(project)


@pytest.mark.parametrize("duration", [-0.1, -1])
def test_negative_durations_are_rejected(duration: float) -> None:
    with pytest.raises(ValueError, match="duration"):
        Shot("shot", duration=duration)
    with pytest.raises(ValueError, match="duration"):
        Scene("scene", "Scene", duration=duration)
