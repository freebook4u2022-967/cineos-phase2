from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.native_video.final_eval import (
    SceneBoundaryEvalReport,
    SceneBoundaryEvidence,
    TemporalFilmEvalReport,
)
from cineos.native_video.production_gate import (
    MeasuredFinalFilmGate,
    plan_scene_boundaries,
)


def _shot(scene_id: str, duration: float, **payload: object) -> SimpleNamespace:
    return SimpleNamespace(scene_id=scene_id, duration=duration, payload=payload)


def _temporal(decision: str = "accept") -> TemporalFilmEvalReport:
    return TemporalFilmEvalReport(
        frame_count=8,
        mean_luma=110.0,
        mean_variance=18.0,
        mean_interframe_mad=6.0,
        black_frame_ratio=0.0,
        frozen_transition_ratio=0.0,
        hard_cut_transition_ratio=0.0,
        decision=decision,
        directives=("repair temporal evidence",) if decision == "reject" else (),
    )


def _boundary_report(decision: str = "accept") -> SceneBoundaryEvalReport:
    evidence = SceneBoundaryEvidence(
        from_scene_id="scene-a",
        to_scene_id="scene-b",
        transition="match",
        outgoing_luma=100.0,
        incoming_luma=105.0,
        boundary_mad=48.0 if decision == "reject" else 5.0,
        decision=decision,
        directives=("repair boundary continuity",) if decision == "reject" else (),
    )
    return SceneBoundaryEvalReport(
        boundary_count=1,
        reject_count=int(decision == "reject"),
        warn_count=int(decision == "warn"),
        mean_boundary_mad=evidence.boundary_mad,
        decision=decision,
        boundaries=(evidence,),
    )


class _TemporalEvaluator:
    def __init__(self, report: TemporalFilmEvalReport) -> None:
        self.report = report
        self.calls = 0

    def evaluate(self, movie_path: object) -> TemporalFilmEvalReport:
        self.calls += 1
        return self.report


class _BoundaryEvaluator:
    def __init__(self, report: SceneBoundaryEvalReport) -> None:
        self.report = report
        self.boundaries = ()

    def evaluate(
        self, movie_path: object, boundaries: object
    ) -> SceneBoundaryEvalReport:
        self.boundaries = tuple(boundaries)
        return self.report


class _UnexpectedBoundaryEvaluator:
    def evaluate(
        self, movie_path: object, boundaries: object
    ) -> SceneBoundaryEvalReport:
        raise AssertionError("single-scene films must not fabricate scene boundaries")


def test_plan_scene_boundaries_uses_cumulative_timeline_and_authored_transition() -> (
    None
):
    plan = (
        _shot("scene-a", 2.0),
        _shot("scene-a", 3.0),
        _shot("scene-b", 4.0, scene_transition="crossfade"),
        _shot("scene-c", 1.0, hard_cut=True),
    )

    boundaries = plan_scene_boundaries(plan)

    assert len(boundaries) == 2
    assert boundaries[0].from_scene_id == "scene-a"
    assert boundaries[0].to_scene_id == "scene-b"
    assert boundaries[0].boundary_seconds == pytest.approx(5.0)
    assert boundaries[0].transition == "fade"
    assert boundaries[1].boundary_seconds == pytest.approx(9.0)
    assert boundaries[1].transition == "cut"


def test_plan_scene_boundaries_defaults_unmarked_scene_change_to_match() -> None:
    boundaries = plan_scene_boundaries((_shot("scene-a", 1.5), _shot("scene-b", 2.0)))
    assert boundaries[0].transition == "match"


def test_serialized_false_flags_do_not_override_authored_transition() -> None:
    boundaries = plan_scene_boundaries(
        (
            _shot("scene-a", 1.0),
            _shot(
                "scene-b",
                1.0,
                transition="crossfade",
                hard_cut="false",
                continuity_reset="0",
            ),
        )
    )

    assert boundaries[0].transition == "fade"


def test_serialized_true_flag_forces_cut() -> None:
    boundaries = plan_scene_boundaries(
        (
            _shot("scene-a", 1.0, transition="fade"),
            _shot("scene-b", 1.0, continuity_reset="yes"),
        )
    )

    assert boundaries[0].transition == "cut"


def test_plan_scene_boundaries_fails_closed_on_ambiguous_boolean_metadata() -> None:
    with pytest.raises(ValueError, match="hard_cut must be boolean metadata"):
        plan_scene_boundaries(
            (_shot("scene-a", 1.0), _shot("scene-b", 1.0, hard_cut="sometimes"))
        )


def test_plan_scene_boundaries_fails_closed_on_unknown_transition() -> None:
    with pytest.raises(ValueError, match="unsupported scene transition"):
        plan_scene_boundaries(
            (_shot("scene-a", 1.0), _shot("scene-b", 1.0, transition="morph"))
        )


def test_measured_final_film_gate_rejects_when_boundary_evidence_rejects(
    tmp_path,
) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"assembled-film")
    temporal = _TemporalEvaluator(_temporal())
    boundary = _BoundaryEvaluator(_boundary_report("reject"))
    gate = MeasuredFinalFilmGate(
        temporal_evaluator=temporal,
        boundary_evaluator=boundary,
    )

    report = gate.evaluate(
        movie,
        (_shot("scene-a", 2.0), _shot("scene-b", 2.0)),
    )

    assert temporal.calls == 1
    assert len(boundary.boundaries) == 1
    assert report.decision == "reject"
    assert report.accepted is False
    assert report.directives == ("repair boundary continuity",)
    assert report.as_dict()["scene_boundaries"]["reject_count"] == 1


def test_measured_final_film_gate_rejects_global_temporal_failure(tmp_path) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"assembled-film")
    gate = MeasuredFinalFilmGate(
        temporal_evaluator=_TemporalEvaluator(_temporal("reject")),
        boundary_evaluator=_BoundaryEvaluator(_boundary_report()),
    )

    report = gate.evaluate(
        movie,
        (_shot("scene-a", 2.0), _shot("scene-b", 2.0)),
    )

    assert report.decision == "reject"
    assert report.directives == ("repair temporal evidence",)


def test_single_scene_still_requires_global_temporal_evidence(tmp_path) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"assembled-film")
    temporal = _TemporalEvaluator(_temporal("accept"))
    gate = MeasuredFinalFilmGate(
        temporal_evaluator=temporal,
        boundary_evaluator=_UnexpectedBoundaryEvaluator(),
    )

    report = gate.evaluate(movie, (_shot("scene-a", 3.0),))

    assert temporal.calls == 1
    assert report.scene_boundaries is None
    assert report.decision == "accept"
