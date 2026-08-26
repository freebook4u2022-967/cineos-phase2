from __future__ import annotations

import json

import pytest

from cineos.native_video.final_audit import (
    AUDIT_RECORD_SHA256_FIELD,
    FINAL_FILM_AUDIT_SCHEMA,
    FinalFilmAuditError,
    FinalFilmAuditRecord,
    load_final_film_audit,
    write_final_film_audit,
)
from cineos.native_video.final_eval import TemporalFilmEvalReport
from cineos.native_video.final_gate import MeasuredFinalFilmReport


def _accepted_report() -> MeasuredFinalFilmReport:
    temporal = TemporalFilmEvalReport(
        frame_count=8,
        mean_luma=96.0,
        mean_variance=42.0,
        mean_interframe_mad=8.0,
        black_frame_ratio=0.0,
        frozen_transition_ratio=0.0,
        hard_cut_transition_ratio=0.125,
        decision="accept",
        directives=(),
    )
    return MeasuredFinalFilmReport(
        decision="accept",
        directives=(),
        temporal=temporal,
    )


def test_final_film_audit_binds_report_to_exact_movie_bytes(tmp_path) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-native-film-v1")

    record = FinalFilmAuditRecord.from_report(
        movie,
        _accepted_report(),
        model_fingerprint="model-sha256:abc",
        runtime_fingerprint="runtime-sha256:def",
    )

    assert record.schema_version == FINAL_FILM_AUDIT_SCHEMA
    assert record.decision == "accept"
    assert record.movie_size_bytes == len(b"cineos-native-film-v1")
    assert record.report["temporal"]["frame_count"] == 8
    payload = record.to_record()
    assert len(payload[AUDIT_RECORD_SHA256_FIELD]) == 64
    record.verify_movie(movie)


def test_final_film_audit_round_trips_and_verifies_movie(tmp_path) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-production-output")
    audit = tmp_path / "qc" / "final-film.json"
    record = FinalFilmAuditRecord.from_report(movie, _accepted_report())

    write_final_film_audit(audit, record, fsync=False)
    loaded = load_final_film_audit(
        audit,
        movie_path=movie,
        require_record_digest=True,
    )

    assert loaded.movie_sha256 == record.movie_sha256
    assert loaded.decision == "accept"
    assert loaded.report["decision"] == "accept"


def test_final_film_audit_detects_movie_mutation(tmp_path) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"original-native-movie")
    record = FinalFilmAuditRecord.from_report(movie, _accepted_report())
    audit = write_final_film_audit(tmp_path / "audit.json", record, fsync=False)

    movie.write_bytes(b"tampered-native-movie")

    with pytest.raises(FinalFilmAuditError, match="digest|size"):
        load_final_film_audit(audit, movie_path=movie)


def test_final_film_audit_rejects_decision_tampering(tmp_path) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-film")
    record = FinalFilmAuditRecord.from_report(movie, _accepted_report())
    audit = write_final_film_audit(tmp_path / "audit.json", record, fsync=False)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["decision"] = "reject"
    audit.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FinalFilmAuditError, match="record digest"):
        load_final_film_audit(audit)


def test_final_film_audit_rejects_report_tampering_even_when_decision_matches(
    tmp_path,
) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-film")
    record = FinalFilmAuditRecord.from_report(movie, _accepted_report())
    audit = write_final_film_audit(tmp_path / "audit.json", record, fsync=False)
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["report"]["temporal"]["frame_count"] = 999
    audit.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FinalFilmAuditError, match="record digest"):
        load_final_film_audit(audit)


def test_final_film_audit_can_read_legacy_v1_but_strict_mode_rejects_it(
    tmp_path,
) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-film")
    record = FinalFilmAuditRecord.from_report(movie, _accepted_report())
    payload = record.to_record()
    payload.pop(AUDIT_RECORD_SHA256_FIELD)
    audit = tmp_path / "legacy-audit.json"
    audit.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_final_film_audit(audit)
    assert loaded.decision == "accept"

    with pytest.raises(FinalFilmAuditError, match="no record integrity digest"):
        load_final_film_audit(audit, require_record_digest=True)


def test_final_film_audit_rejects_unknown_schema(tmp_path) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-film")
    record = FinalFilmAuditRecord.from_report(movie, _accepted_report())
    payload = record.to_record()
    payload["schema_version"] = "cineos.native_video.final_film_audit.v999"
    payload.pop(AUDIT_RECORD_SHA256_FIELD)
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FinalFilmAuditError, match="invalid final-film audit"):
        load_final_film_audit(audit)


def test_final_film_audit_refuses_empty_movie(tmp_path) -> None:
    movie = tmp_path / "empty.mp4"
    movie.write_bytes(b"")

    with pytest.raises(FinalFilmAuditError, match="empty movie"):
        FinalFilmAuditRecord.from_report(movie, _accepted_report())
