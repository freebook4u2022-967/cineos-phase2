"""NOVA Director Alpha integration tests."""

from dataclasses import asdict

import pytest

from cineos.assets import AssetRegistry
from cineos.assets import Character as AssetCharacter
from cineos.assets import Environment as AssetEnvironment
from cineos.compiler import FilmCompiler
from cineos.nova import (
    CreativeBrief,
    MissingAssetError,
    NOVACritic,
    NOVADirector,
    NOVARevisionEngine,
)
from cineos.nova.serializer import serialize


def setup_plan():
    registry = AssetRegistry()
    character = registry.register(AssetCharacter(name="Ari"))
    environment = registry.register(AssetEnvironment(name="Observatory"))
    brief = CreativeBrief(
        "Signal",
        "answer a signal before dawn",
        theme="connection",
        target_duration=30,
        required_characters=[str(character.asset_id)],
        required_environments=[str(environment.asset_id)],
    )
    return NOVADirector(registry), brief


def test_rule_planner_is_deterministic_and_compiles():
    director, brief = setup_plan()
    first = director.create_plan(brief, seed=42)
    second = director.create_plan(brief, seed=42)
    assert serialize(first) == serialize(second)
    assert sum(shot.duration for shot in first.shots) == brief.target_duration
    assert FilmCompiler().compile(first.project).shot_manifest
    assert first.story.content_hash == second.story.content_hash


def test_continuity_and_targeted_revision_preserve_ids():
    director, brief = setup_plan()
    plan = director.create_plan(brief)
    assert plan.scenes[1].continuity_inputs == plan.scenes[0].continuity_outputs
    plan.shots[1].framing = plan.shots[0].framing
    plan.shots[1].camera_movement = plan.shots[0].camera_movement
    finding = NOVACritic().critique(plan)[0]
    unchanged = asdict(plan.shots[-1])
    revised = NOVARevisionEngine().revise(plan, [finding])
    assert [item.shot_id for item in revised.shots] == [
        item.shot_id for item in plan.shots
    ]
    assert asdict(revised.shots[-1]) == unchanged
    assert revised.revision_history


def test_missing_approved_asset_stops_planning():
    brief = CreativeBrief("No", "find someone", required_characters=["unknown"])
    with pytest.raises(MissingAssetError):
        NOVADirector().create_plan(brief)
