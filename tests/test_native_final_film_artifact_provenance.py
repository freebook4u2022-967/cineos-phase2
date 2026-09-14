from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from cineos.native_video.artifact_integrity import ArtifactIntegrityError
from cineos.native_video.duration_gate import DurationIntegrityReport
from cineos.native_video.final_eval import TemporalFilmEvalReport
from cineos.native_video.final_gate import MeasuredFinalFilmGate


class _TemporalEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, movie_path):
        self.calls += 1
        return TemporalFilmEvalReport(
            frame_count=2,
            mean_luma=96.0,
            mean_variance=12.0,
            mean_interframe_mad=8.0,
            black_frame_ratio=0.0,
            frozen_transition_ratio=0.0,
            hard_cut_transition_ratio=0.0,
            decision="accept",
            directives=(),
        )


class _DurationEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, movie_path, plan):
        self.calls += 1
        return DurationIntegrityReport(
            planned_seconds=1.0,
            measured_seconds=1.0,
            delta_seconds=0.0,
            allowed_error_seconds=0.25,
            decision="accept",
            directives=(),
        )


class _BoundaryEvaluator:
    def evaluate(self, movie_path, boundaries):
        raise AssertionError("single-scene plan must not require boundary evaluation")


def _plan():
    return [
        SimpleNamespace(
            shot_id="shot-001",
            scene_id="scene-001",
            index=0,
            duration=1.0,
            payload={},
        )
    ]


def test_final_film_report_binds_qc_to_exact_artifact_bytes(tmp_path):
    movie = tmp_path / "movie.mp4"
    payload = b"cineos-final-film-artifact"
    movie.write_bytes(payload)

    temporal = _TemporalEvaluator()
    duration = _DurationEvaluator()
    gate = MeasuredFinalFilmGate(
        temporal_evaluator=temporal,
        duration_evaluator=duration,
        boundary_evaluator=_BoundaryEvaluator(),
    )

    report = gate.evaluate(movie, _plan())

    assert report.accepted
    assert report.artifact.byte_size == len(payload)
    assert report.artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert report.as_dict()["artifact"] == {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_size": len(payload),
    }
    assert temporal.calls == 1
    assert duration.calls == 1


def test_final_film_gate_fails_closed_on_empty_artifact_before_quality_evaluators(
    tmp_path,
):
    movie = tmp_path / "empty.mp4"
    movie.write_bytes(b"")
    temporal = _TemporalEvaluator()
    duration = _DurationEvaluator()
    gate = MeasuredFinalFilmGate(
        temporal_evaluator=temporal,
        duration_evaluator=duration,
        boundary_evaluator=_BoundaryEvaluator(),
    )

    with pytest.raises(ArtifactIntegrityError, match="empty artifact"):
        gate.evaluate(movie, _plan())

    assert temporal.calls == 0
    assert duration.calls == 0
