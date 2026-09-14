from __future__ import annotations

import pytest

from cineos.native_video.artifact_integrity import provenance_for
from cineos.native_video.final_audit import FinalFilmAuditError, FinalFilmAuditRecord
from cineos.native_video.final_eval import TemporalFilmEvalReport
from cineos.native_video.final_gate import MeasuredFinalFilmReport


def _accepted_report(movie_path) -> MeasuredFinalFilmReport:
    return MeasuredFinalFilmReport(
        decision="accept",
        directives=(),
        temporal=TemporalFilmEvalReport(
            frame_count=4,
            mean_luma=96.0,
            mean_variance=24.0,
            mean_interframe_mad=6.0,
            black_frame_ratio=0.0,
            frozen_transition_ratio=0.0,
            hard_cut_transition_ratio=0.0,
            decision="accept",
            directives=(),
        ),
        artifact=provenance_for(movie_path),
    )


def test_final_film_audit_accepts_report_for_exact_movie_bytes(tmp_path) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-final-film")
    report = _accepted_report(movie)

    record = FinalFilmAuditRecord.from_report(movie, report)

    assert record.movie_sha256 == report.artifact.sha256
    assert record.movie_size_bytes == report.artifact.byte_size


def test_final_film_audit_rejects_replayed_report_for_different_movie(tmp_path) -> None:
    evaluated = tmp_path / "evaluated.mp4"
    evaluated.write_bytes(b"cineos-evaluated-film")
    report = _accepted_report(evaluated)

    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(b"cineos-different-film")

    with pytest.raises(FinalFilmAuditError, match="artifact digest"):
        FinalFilmAuditRecord.from_report(replacement, report)


def test_final_film_audit_rejects_replayed_report_when_size_changes(tmp_path) -> None:
    evaluated = tmp_path / "evaluated.mp4"
    evaluated.write_bytes(b"cineos-evaluated-film")
    report = _accepted_report(evaluated)

    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(b"short")

    with pytest.raises(FinalFilmAuditError, match="artifact size"):
        FinalFilmAuditRecord.from_report(replacement, report)
