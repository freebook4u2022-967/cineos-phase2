from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from cineos.native_video.release_record import (
    FinalFilmReleaseRecord,
    FinalFilmReleaseRecordError,
    build_release_record,
    verify_release_record,
)
from cineos.native_video.runtime_manifest import ProductionRuntimeManifest


@dataclass(frozen=True)
class _Shot:
    shot_id: str
    duration_seconds: float


@dataclass(frozen=True)
class _QualityReport:
    decision: str
    metric: float

    def as_dict(self) -> dict[str, object]:
        return {"decision": self.decision, "metric": self.metric}


def _runtime_manifest() -> ProductionRuntimeManifest:
    return ProductionRuntimeManifest(
        renderer_id="cineos-native",
        temporal_model_fingerprint="temporal-v1",
        device="cpu",
        max_recovery_attempts=2,
        require_final_film_evaluation=True,
        require_audio=True,
        final_gate_policy_fingerprint="gate-policy-v1",
        native_model_manifest_sha256="model-release-v1",
    )


def test_release_record_round_trip_and_verification(tmp_path: Path) -> None:
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"cineos-final-film-bytes")
    plan = [_Shot("s1", 1.5), _Shot("s2", 2.0)]
    quality = _QualityReport("accept", 0.98)

    record = build_release_record(
        movie,
        runtime_manifest=_runtime_manifest(),
        plan=plan,
        quality_report=quality,
    )
    restored = FinalFilmReleaseRecord.restore(record.snapshot())

    provenance = verify_release_record(
        restored,
        movie,
        runtime_manifest=_runtime_manifest(),
        plan=plan,
        quality_report=quality,
    )

    assert provenance.sha256 == record.artifact_sha256
    assert provenance.byte_size == record.artifact_bytes
    assert record.decision == "accept"


def test_release_record_rejects_changed_movie_bytes(tmp_path: Path) -> None:
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"original-film")
    plan = [_Shot("s1", 1.0)]
    quality = _QualityReport("accept", 0.95)
    record = build_release_record(
        movie,
        runtime_manifest=_runtime_manifest(),
        plan=plan,
        quality_report=quality,
    )

    movie.write_bytes(b"tampered-film")

    with pytest.raises(FinalFilmReleaseRecordError):
        verify_release_record(
            record,
            movie,
            runtime_manifest=_runtime_manifest(),
            plan=plan,
            quality_report=quality,
        )


def test_release_record_rejects_changed_plan_or_quality_evidence(tmp_path: Path) -> None:
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"film")
    plan = [_Shot("s1", 1.0)]
    quality = _QualityReport("warn", 0.85)
    record = build_release_record(
        movie,
        runtime_manifest=_runtime_manifest(),
        plan=plan,
        quality_report=quality,
    )

    with pytest.raises(FinalFilmReleaseRecordError, match="plan_sha256"):
        verify_release_record(
            record,
            movie,
            runtime_manifest=_runtime_manifest(),
            plan=[_Shot("s1", 1.25)],
            quality_report=quality,
        )

    with pytest.raises(FinalFilmReleaseRecordError, match="quality_report_sha256"):
        verify_release_record(
            record,
            movie,
            runtime_manifest=_runtime_manifest(),
            plan=plan,
            quality_report=_QualityReport("warn", 0.80),
        )


def test_release_record_rejects_runtime_or_model_policy_change(tmp_path: Path) -> None:
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"film")
    plan = [_Shot("s1", 1.0)]
    quality = _QualityReport("accept", 0.99)
    record = build_release_record(
        movie,
        runtime_manifest=_runtime_manifest(),
        plan=plan,
        quality_report=quality,
    )
    changed = ProductionRuntimeManifest(
        renderer_id="cineos-native",
        temporal_model_fingerprint="temporal-v2",
        device="cpu",
        max_recovery_attempts=2,
        require_final_film_evaluation=True,
        require_audio=True,
        final_gate_policy_fingerprint="gate-policy-v2",
        native_model_manifest_sha256="model-release-v2",
    )

    with pytest.raises(FinalFilmReleaseRecordError, match="runtime_manifest_sha256"):
        verify_release_record(
            record,
            movie,
            runtime_manifest=changed,
            plan=plan,
            quality_report=quality,
        )


def test_release_record_refuses_rejected_quality_report(tmp_path: Path) -> None:
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"film")

    with pytest.raises(FinalFilmReleaseRecordError, match="without accepted quality"):
        build_release_record(
            movie,
            runtime_manifest=_runtime_manifest(),
            plan=[_Shot("s1", 1.0)],
            quality_report=_QualityReport("reject", 0.1),
        )


def test_restore_rejects_unknown_schema_and_fields(tmp_path: Path) -> None:
    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"film")
    record = build_release_record(
        movie,
        runtime_manifest=_runtime_manifest(),
        plan=[_Shot("s1", 1.0)],
        quality_report=_QualityReport("accept", 1.0),
    )

    wrong_schema = record.snapshot()
    wrong_schema["schema"] = "cineos-final-film-release/9.9"
    with pytest.raises(FinalFilmReleaseRecordError, match="unsupported"):
        FinalFilmReleaseRecord.restore(wrong_schema)

    unknown_field = record.snapshot()
    unknown_field["surprise"] = True
    with pytest.raises(FinalFilmReleaseRecordError, match="unknown fields"):
        FinalFilmReleaseRecord.restore(unknown_field)
