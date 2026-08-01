from pathlib import Path

import pytest

from cineos.atlas import Range, RendererCapabilities, Resolution
from cineos.cli.main import main
from cineos.conditioning import (
    CameraConditioning,
    ConditioningPackage,
    ConditioningValidator,
    ContinuityConditioning,
    RendererCapabilityRequirements,
    UnsupportedRendererCapabilities,
    deserialize,
    serialize,
)
from cineos.conditioning.serializer import calculate_content_hash, save


def package() -> ConditioningPackage:
    value = ConditioningPackage(
        shot_id="shot-1",
        scene_id="scene-1",
        character_conditioning=[],
        environment_conditioning=None,
        wardrobe_conditioning=[],
        prop_conditioning=[],
        camera_conditioning=CameraConditioning(
            resolution=(1920, 1080), fps=24, duration=2
        ),
        continuity_constraints=ContinuityConditioning(),
        approved_reference_ids=["approved-reference"],
        renderer_capability_requirements=RendererCapabilityRequirements(
            image_reference_support=True,
            maximum_duration=2,
            supported_resolution=(1920, 1080),
            supported_fps=24,
        ),
        deterministic_seed=42,
    )
    value.content_hash = calculate_content_hash(value)
    return value


def test_serialization_is_deterministic_and_round_trips():
    first = package()
    encoded = serialize(first)
    second = deserialize(encoded)
    assert serialize(second) == encoded
    assert second.camera_conditioning.resolution == (1920, 1080)
    assert ConditioningValidator().validate(second) == []


def test_renderer_capabilities_are_checked_before_execution():
    capabilities = RendererCapabilities(
        (Resolution(1920, 1080),), Range(0, 10), (24,), frozenset()
    )
    with pytest.raises(UnsupportedRendererCapabilities, match="image-reference"):
        ConditioningValidator().validate_renderer(package(), capabilities)


def test_renderer_accepts_supported_conditioning_features():
    capabilities = RendererCapabilities(
        (Resolution(1920, 1080),),
        Range(0, 10),
        (24,),
        frozenset({"image-reference"}),
    )
    ConditioningValidator().validate_renderer(package(), capabilities)


def test_content_hash_detects_package_edits():
    value = package()
    value.shot_id = "edited"
    assert "content hash" in ";".join(ConditioningValidator().validate(value))


def test_condition_validate_cli(tmp_path: Path, capsys):
    path = tmp_path / "conditioning.json"
    save(package(), path)
    assert main(["condition", "validate", str(path)]) == 0
    assert "is valid" in capsys.readouterr().out
