from __future__ import annotations

from pathlib import Path

from cineos.film.planner import PlannedShot
from cineos.native_video.audio_integrity import (
    AudioIntegrityReport,
    AudioStreamEvidence,
)
from cineos.native_video.duration_gate import DurationIntegrityReport
from cineos.native_video.final_eval import TemporalFilmEvalReport
from cineos.native_video.final_gate import MeasuredFinalFilmGate


class _TemporalEvaluator:
    def evaluate(self, movie_path: str | Path) -> TemporalFilmEvalReport:
        return TemporalFilmEvalReport(
            frame_count=8,
            mean_luma=96.0,
            mean_variance=12.0,
            mean_interframe_mad=4.0,
            black_frame_ratio=0.0,
            frozen_transition_ratio=0.0,
            hard_cut_transition_ratio=0.0,
            decision="accept",
            directives=(),
        )


class _DurationEvaluator:
    def evaluate(self, movie_path: str | Path, plan) -> DurationIntegrityReport:
        planned = sum(float(shot.duration) for shot in plan)
        return DurationIntegrityReport(
            planned_seconds=planned,
            measured_seconds=planned,
            delta_seconds=0.0,
            allowed_error_seconds=0.25,
            decision="accept",
            directives=(),
        )


class _UnexpectedBoundaryEvaluator:
    def evaluate(self, movie_path, boundaries):
        raise AssertionError("single-scene test film must not invoke boundary QC")


class _AudioEvaluator:
    def __init__(self, decision: str = "accept") -> None:
        self.decision = decision
        self.calls: list[tuple[Path, float | None, bool]] = []

    def evaluate(
        self,
        movie_path: str | Path,
        *,
        expected_duration_seconds: float | None = None,
        required: bool = True,
    ) -> AudioIntegrityReport:
        self.calls.append((Path(movie_path), expected_duration_seconds, required))
        directives = (
            ("restore or render the required final-film audio stream",)
            if self.decision == "reject"
            else ()
        )
        stream = None
        if self.decision != "reject":
            stream = AudioStreamEvidence(
                codec_name="aac",
                sample_rate_hz=48000,
                channels=2,
                duration_seconds=float(expected_duration_seconds or 1.0),
            )
        return AudioIntegrityReport(
            decision=self.decision,
            required=required,
            stream=stream,
            expected_duration_seconds=expected_duration_seconds,
            duration_delta_seconds=0.0 if stream is not None else None,
            directives=directives,
        )


class _NeverAudioEvaluator:
    def evaluate(self, *args, **kwargs):
        raise AssertionError("audio QC must remain disabled for legacy silent builds")


def _plan() -> list[PlannedShot]:
    return [PlannedShot("a1", "scene-a", 2.0, 0, {})]


def _gate(*, require_audio: bool, audio_evaluator) -> MeasuredFinalFilmGate:
    return MeasuredFinalFilmGate(
        temporal_evaluator=_TemporalEvaluator(),
        boundary_evaluator=_UnexpectedBoundaryEvaluator(),
        duration_evaluator=_DurationEvaluator(),
        audio_evaluator=audio_evaluator,
        require_audio=require_audio,
    )


def test_required_audio_rejection_fails_entire_final_film_gate(tmp_path) -> None:
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"movie")
    audio = _AudioEvaluator("reject")
    gate = _gate(require_audio=True, audio_evaluator=audio)

    report = gate.evaluate(movie, _plan())

    assert report.decision == "reject"
    assert report.audio is not None
    assert report.audio.decision == "reject"
    assert report.directives == (
        "restore or render the required final-film audio stream",
    )
    assert audio.calls == [(movie, 2.0, True)]
    assert report.as_dict()["audio"]["decision"] == "reject"


def test_healthy_required_audio_participates_in_acceptance(tmp_path) -> None:
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"movie")
    audio = _AudioEvaluator("accept")
    gate = _gate(require_audio=True, audio_evaluator=audio)

    report = gate.evaluate(movie, _plan())

    assert report.decision == "accept"
    assert report.audio is not None
    assert report.audio.accepted
    assert report.audio.expected_duration_seconds == 2.0
    assert audio.calls == [(movie, 2.0, True)]


def test_legacy_silent_gate_does_not_invoke_audio_evaluator(tmp_path) -> None:
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"movie")
    gate = _gate(require_audio=False, audio_evaluator=_NeverAudioEvaluator())

    report = gate.evaluate(movie, _plan())

    assert report.decision == "accept"
    assert report.audio is None
    assert report.as_dict()["audio"] is None
