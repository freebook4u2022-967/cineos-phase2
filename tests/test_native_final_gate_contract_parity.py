from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.native_video.final_gate import _planned_scene_boundaries
from cineos.native_video.production_gate import plan_scene_boundaries


def _shot(scene_id: str, duration: float, **payload: object) -> SimpleNamespace:
    return SimpleNamespace(scene_id=scene_id, duration=duration, payload=payload)


def _normalized(boundaries: object) -> tuple[tuple[str, str, float, str], ...]:
    return tuple(
        (
            item.from_scene_id,
            item.to_scene_id,
            item.boundary_seconds,
            item.transition,
        )
        for item in boundaries
    )


@pytest.mark.parametrize(
    "plan",
    [
        (
            _shot("scene-a", 1.0),
            _shot("scene-b", 2.0),
        ),
        (
            _shot("scene-a", 1.0, transition_out="match_cut"),
            _shot("scene-b", 2.0),
        ),
        (
            _shot("scene-a", 1.0, transition_out="match"),
            _shot("scene-b", 2.0, transition_in="crossfade"),
        ),
        (
            _shot("scene-a", 1.0, transition_out="fade"),
            _shot(
                "scene-b",
                2.0,
                transition_in="match",
                continuity_reset="true",
            ),
        ),
        (
            _shot("scene-a", 1.0),
            _shot("scene-a", 2.0),
            _shot("scene-b", 3.0, scene_transition="cross_fade"),
            _shot("scene-c", 4.0, hard_cut=1),
        ),
    ],
)
def test_final_film_gate_paths_share_the_same_authored_edit_contract(plan: tuple[object, ...]) -> None:
    """Legacy and canonical final gates must interpret the same shot plan identically.

    Both gates remain public while the production path migrates toward the richer
    canonical final gate. This regression prevents a persisted plan from being
    accepted by one path and rejected by the other solely because transition
    aliases, serialized boolean flags, or cumulative timestamps drift apart.
    """

    legacy = plan_scene_boundaries(plan)
    canonical = _planned_scene_boundaries(plan)

    assert _normalized(legacy) == _normalized(canonical)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"hard_cut": "sometimes"}, "hard_cut must be boolean metadata"),
        ({"continuity_reset": "maybe"}, "continuity_reset must be boolean metadata"),
        ({"transition": "morph"}, "unsupported"),
    ],
)
def test_final_film_gate_paths_fail_closed_on_the_same_invalid_edit_metadata(
    payload: dict[str, object], message: str
) -> None:
    plan = (_shot("scene-a", 1.0), _shot("scene-b", 2.0, **payload))

    with pytest.raises(ValueError, match=message):
        plan_scene_boundaries(plan)
    with pytest.raises(ValueError, match=message):
        _planned_scene_boundaries(plan)


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf")])
def test_final_film_gate_paths_fail_closed_on_the_same_invalid_timeline(
    duration: float,
) -> None:
    plan = (_shot("scene-a", duration), _shot("scene-b", 1.0))

    with pytest.raises(ValueError, match="finite and positive"):
        plan_scene_boundaries(plan)
    with pytest.raises(ValueError, match="finite and positive"):
        _planned_scene_boundaries(plan)
