import json

import pytest

from cineos.short_drama import DramaBrief, ShortDramaOrchestrator
from cineos.short_drama.cli import create_drama_plan, write_drama_plan


def test_one_line_brief_becomes_continuity_checked_plan():
    brief = DramaBrief(
        premise="A man receives a message from his wife who died three years ago.",
        duration_seconds=180,
        genre="mystery",
        tone="tense and intimate",
    )

    plan = ShortDramaOrchestrator().plan(brief)

    assert plan.story["premise"] == brief.premise
    assert plan.story["theme"] == "truth has a personal cost"
    assert len(plan.characters) == 2
    assert len(plan.screenplay["beats"]) == 5
    assert len(plan.screenplay["scenes"]) == 5
    assert len(plan.direction["decisions"]) == 5
    assert len(plan.scene_states) == 6
    assert len(plan.shots) == 5
    assert sum(shot["duration_seconds"] for shot in plan.shots) == pytest.approx(180)
    assert plan.shots[2]["shot_size"] == "close-up"
    assert plan.shots[2]["lens"] == "85mm"
    assert plan.continuity["status"] == "pass"
    assert plan.continuity["shot_order"] == [shot["shot_id"] for shot in plan.shots]
    assert plan.continuity["scene_state_count"] == 6


def test_drama_create_adapter_returns_json_safe_package():
    payload = create_drama_plan(
        "A man receives a message from his wife who died three years ago.",
        duration_seconds=90,
        genre="mystery",
        tone="intimate",
    )

    encoded = json.dumps(payload)
    assert encoded
    assert payload["brief"]["duration_seconds"] == 90
    assert payload["characters"][0]["character_id"] == "char-protagonist"
    assert payload["direction"]["decisions"][0]["camera_movement"] == "slow push-in"


def test_write_drama_plan_persists_package(tmp_path):
    destination = tmp_path / "drama-package.json"
    write_drama_plan("A stranger leaves a key at midnight.", destination)

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["story"]["premise"].startswith("A stranger")
    assert payload["shots"]


def test_empty_premise_is_rejected():
    with pytest.raises(ValueError, match="premise"):
        DramaBrief(premise="   ")
