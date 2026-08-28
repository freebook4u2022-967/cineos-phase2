from types import SimpleNamespace

from cineos.film.planner import PlannedShot
from cineos.native_video.final_repair import build_final_film_repair_plan


def _component(decision="accept", directives=()):
    return SimpleNamespace(decision=decision, directives=tuple(directives))


def _plan():
    return (
        PlannedShot("a1", "scene-a", 2.0, 0, {}),
        PlannedShot("a2", "scene-a", 2.0, 1, {}),
        PlannedShot("b1", "scene-b", 2.0, 2, {}),
        PlannedShot("b2", "scene-b", 2.0, 3, {}),
    )


def test_boundary_repair_targets_only_adjacent_scene_edge_shots():
    boundary = SimpleNamespace(
        from_scene_id="scene-a",
        to_scene_id="scene-b",
        decision="reject",
        directives=("repair measured match-boundary drift",),
    )
    boundaries = SimpleNamespace(boundaries=(boundary,))

    repair = build_final_film_repair_plan(
        plan=_plan(),
        temporal=_component(),
        boundaries=boundaries,
        duration=_component(),
    )

    assert repair.required
    assert repair.affected_shot_ids == ("a2", "b1")
    assert len(repair.actions) == 1
    assert repair.actions[0].domain == "scene_continuity"
    assert repair.actions[0].scene_ids == ("scene-a", "scene-b")
    assert repair.actions[0].shot_ids == ("a2", "b1")


def test_global_temporal_rejection_does_not_guess_a_bad_shot():
    repair = build_final_film_repair_plan(
        plan=_plan(),
        temporal=_component("reject", ("rerender frozen temporal regions",)),
        duration=_component(),
    )

    assert repair.required
    assert repair.affected_shot_ids == ()
    assert repair.actions[0].domain == "visual_timeline"
    assert repair.actions[0].shot_ids == ()


def test_assembly_and_audio_rejections_preserve_healthy_visual_shots():
    repair = build_final_film_repair_plan(
        plan=_plan(),
        temporal=_component(),
        duration=_component("reject", ("rebuild final assembly",)),
        audio=_component("reject", ("remux damaged audio stream",)),
    )

    assert repair.required
    assert repair.affected_shot_ids == ()
    assert [action.domain for action in repair.actions] == ["assembly", "audio"]


def test_accepted_film_has_empty_repair_plan():
    repair = build_final_film_repair_plan(
        plan=_plan(),
        temporal=_component(),
        duration=_component(),
        audio=_component(),
    )

    assert not repair.required
    assert repair.actions == ()
    assert repair.affected_shot_ids == ()


def test_duplicate_repair_evidence_is_deduplicated_stably():
    repair = build_final_film_repair_plan(
        plan=_plan(),
        temporal=_component(
            "reject", ("repair temporal defect", "repair temporal defect")
        ),
        duration=_component(),
    )

    assert len(repair.actions) == 1
    assert repair.actions[0].reason == "repair temporal defect"
