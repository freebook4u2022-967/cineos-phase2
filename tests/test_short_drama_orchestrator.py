import pytest

from cineos.short_drama import DramaBrief, ShortDramaOrchestrator


def test_one_line_brief_becomes_continuity_checked_plan():
    brief = DramaBrief(
        premise="A man receives a message from his wife who died three years ago.",
        duration_seconds=180,
        genre="mystery",
        tone="tense and intimate",
    )

    plan = ShortDramaOrchestrator().plan(brief)

    assert plan.story["premise"] == brief.premise
    assert len(plan.screenplay["beats"]) == 5
    assert len(plan.shots) == 5
    assert sum(shot["duration_seconds"] for shot in plan.shots) == pytest.approx(180)
    assert plan.continuity["status"] == "pass"
    assert plan.continuity["shot_order"] == [shot["shot_id"] for shot in plan.shots]


def test_empty_premise_is_rejected():
    with pytest.raises(ValueError, match="premise"):
        DramaBrief(premise="   ")
