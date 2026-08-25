from cineos.film.planner import PlannedShot
from cineos.native_video.final_eval import (
    SceneBoundaryEvalReport,
    SceneBoundaryEvidence,
    TemporalFilmEvalReport,
)
from cineos.native_video.final_gate import MeasuredFinalFilmGate, _planned_scene_boundaries


class _TemporalEvaluator:
    def __init__(self, decision="accept", directives=()):
        self.report = TemporalFilmEvalReport(
            frame_count=8,
            mean_luma=96.0,
            mean_variance=12.0,
            mean_interframe_mad=4.0,
            black_frame_ratio=0.0,
            frozen_transition_ratio=0.0,
            hard_cut_transition_ratio=0.0,
            decision=decision,
            directives=tuple(directives),
        )
        self.calls = []

    def evaluate(self, movie_path):
        self.calls.append(movie_path)
        return self.report


class _BoundaryEvaluator:
    def __init__(self, decision="accept", directives=()):
        evidence = SceneBoundaryEvidence(
            from_scene_id="scene-a",
            to_scene_id="scene-b",
            transition="match",
            outgoing_luma=90.0,
            incoming_luma=91.0,
            boundary_mad=2.0,
            decision=decision,
            directives=tuple(directives),
        )
        self.report = SceneBoundaryEvalReport(
            boundary_count=1,
            reject_count=int(decision == "reject"),
            warn_count=int(decision == "warn"),
            mean_boundary_mad=2.0,
            decision=decision,
            boundaries=(evidence,),
        )
        self.calls = []

    def evaluate(self, movie_path, boundaries):
        self.calls.append((movie_path, tuple(boundaries)))
        return self.report


def _plan():
    return [
        PlannedShot("a1", "scene-a", 2.0, 0, {}),
        PlannedShot("a2", "scene-a", 3.0, 1, {"transition_out": "fade"}),
        PlannedShot("b1", "scene-b", 4.0, 2, {"transition_in": "match"}),
    ]


def test_plan_boundaries_use_elapsed_duration_and_explicit_transition():
    boundaries = _planned_scene_boundaries(_plan())

    assert len(boundaries) == 1
    assert boundaries[0].from_scene_id == "scene-a"
    assert boundaries[0].to_scene_id == "scene-b"
    assert boundaries[0].boundary_seconds == 5.0
    assert boundaries[0].transition == "match"


def test_single_scene_skips_boundary_decoder(tmp_path):
    temporal = _TemporalEvaluator()
    boundary = _BoundaryEvaluator()
    gate = MeasuredFinalFilmGate(temporal, boundary)
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"movie")
    plan = [PlannedShot("a1", "scene-a", 2.0, 0, {})]

    report = gate.evaluate(movie, plan)

    assert report.decision == "accept"
    assert len(temporal.calls) == 1
    assert boundary.calls == []
    assert report.boundaries is None


def test_boundary_rejection_overrides_temporal_acceptance(tmp_path):
    temporal = _TemporalEvaluator("accept")
    boundary = _BoundaryEvaluator("reject", ("repair match boundary",))
    gate = MeasuredFinalFilmGate(temporal, boundary)
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"movie")

    report = gate.evaluate(movie, _plan())

    assert report.decision == "reject"
    assert report.directives == ("repair match boundary",)
    assert boundary.calls[0][1][0].boundary_seconds == 5.0
    assert report.as_dict()["boundaries"]["decision"] == "reject"


def test_temporal_warning_and_boundary_directives_are_deduplicated(tmp_path):
    temporal = _TemporalEvaluator("warn", ("review drift", "review drift"))
    boundary = _BoundaryEvaluator("warn", ("review drift", "inspect edit"))
    gate = MeasuredFinalFilmGate(temporal, boundary)
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"movie")

    report = gate.evaluate(movie, _plan())

    assert report.decision == "warn"
    assert report.directives == ("review drift", "inspect edit")


def test_invalid_planned_transition_fails_closed(tmp_path):
    temporal = _TemporalEvaluator()
    boundary = _BoundaryEvaluator()
    gate = MeasuredFinalFilmGate(temporal, boundary)
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"movie")
    plan = [
        PlannedShot("a", "scene-a", 1.0, 0, {}),
        PlannedShot("b", "scene-b", 1.0, 1, {"transition_in": "teleport"}),
    ]

    try:
        gate.evaluate(movie, plan)
    except ValueError as error:
        assert "unsupported planned scene transition" in str(error)
    else:
        raise AssertionError("invalid transition contract must fail closed")
