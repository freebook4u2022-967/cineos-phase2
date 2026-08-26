from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.native_video.final_gate import _planned_scene_boundaries
from cineos.native_video.production_gate import plan_scene_boundaries


def _shot(scene_id: str, duration: float, **payload: object) -> SimpleNamespace:
    return SimpleNamespace(scene_id=scene_id, duration=duration, payload=payload)


def _signature(boundaries: object) -> tuple[tuple[str, str, float, str], ...]:
    return tuple(
        (
            boundary.from_scene_id,
            boundary.to_scene_id,
            boundary.boundary_seconds,
            boundary.transition,
        )
        for boundary in boundaries
    )


@pytest.mark.parametrize(
    "plan",
    [
        (_shot("a", 1.0), _shot("b", 2.0)),
        (
            _shot("a", 1.0, transition_out="match_cut"),
            _shot("b", 2.0),
        ),
        (
            _shot("a", 1.0, transition_out="match"),
            _shot("b", 2.0, transition_in="crossfade"),
        ),
        (
            _shot("a", 1.0),
            _shot("b", 2.0, transition="fade", hard_cut="false"),
            _shot("c", 3.0, continuity_reset="yes"),
        ),
        (
            _shot("a", 1.0),
            _shot("a", 2.0),
            _shot("b", 3.0, scene_transition="match-cut"),
        ),
    ],
)
def test_final_film_gate_paths_share_scene_boundary_contract(plan: tuple[object, ...]) -> None:
    """Both public/legacy final-film paths must interpret authored edits identically.

    CINEOS currently exposes a richer production final gate while retaining the
    earlier production_gate API for compatibility. Until the compatibility path is
    fully retired, drift in transition parsing would make identical film plans pass
    one acceptance path and fail another. This regression locks the contract.
    """

    assert _signature(_planned_scene_boundaries(plan)) == _signature(
        plan_scene_boundaries(plan)
    )


@pytest.mark.parametrize(
    "plan",
    [
        (_shot("a", 1.0), _shot("b", 1.0, hard_cut="sometimes")),
        (_shot("a", 1.0), _shot("b", 1.0, transition="morph")),
        (_shot("a", float("nan")), _shot("b", 1.0)),
        (_shot("a", float("inf")), _shot("b", 1.0)),
    ],
)
def test_final_film_gate_paths_fail_closed_on_same_invalid_plans(
    plan: tuple[object, ...],
) -> None:
    errors: list[type[BaseException]] = []
    for planner in (_planned_scene_boundaries, plan_scene_boundaries):
        try:
            planner(plan)
        except BaseException as exc:  # noqa: BLE001 - parity test records exact failure class.
            errors.append(type(exc))
        else:
            pytest.fail(f"{planner.__module__}.{planner.__name__} accepted invalid plan")

    assert errors == [ValueError, ValueError]
