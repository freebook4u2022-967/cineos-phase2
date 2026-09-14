from __future__ import annotations

import json

import pytest

from cineos.native_video.artifact_integrity import provenance_for
from cineos.native_video.audited_release import (
    AuditedProductionRelease,
    create_audited_production_release,
    load_audited_production_release,
    save_audited_production_release,
    verify_audited_production_release,
)
from cineos.native_video.final_audit import (
    FinalFilmAuditRecord,
    write_final_film_audit,
)
from cineos.native_video.final_eval import TemporalFilmEvalReport
from cineos.native_video.final_gate import MeasuredFinalFilmReport
from cineos.native_video.release_bundle import ProductionReleaseBundle
from cineos.native_video.release_receipt import ProductionReleaseError
from cineos.native_video.runtime_manifest import ProductionRuntimeManifest


def _runtime(*, model_digest: str = "a" * 64) -> ProductionRuntimeManifest:
    return ProductionRuntimeManifest(
        renderer_id="cineos-native-video",
        temporal_model_fingerprint="temporal-v1",
        device="cpu",
        max_recovery_attempts=2,
        require_final_film_evaluation=True,
        require_audio=True,
        final_gate_policy_fingerprint="b" * 64,
        native_model_manifest_sha256=model_digest,
    )


def _report(movie_path) -> MeasuredFinalFilmReport:
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


def _bundle(runtime: ProductionRuntimeManifest) -> ProductionReleaseBundle:
    return ProductionReleaseBundle(
        receipt_sha256="c" * 64,
        readiness_attestation_fingerprint="d" * 64,
        runtime_manifest_fingerprint=runtime.fingerprint,
    )


def _audit(tmp_path, runtime: ProductionRuntimeManifest):
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-audited-production-film")
    record = FinalFilmAuditRecord.from_production_runtime(
        movie,
        _report(movie),
        runtime,
    )
    audit = tmp_path / "film.audit.json"
    write_final_film_audit(audit, record, fsync=False)
    return movie, audit


def test_audited_release_binds_bundle_audit_movie_model_and_runtime(tmp_path) -> None:
    runtime = _runtime()
    bundle = _bundle(runtime)
    movie, audit = _audit(tmp_path, runtime)

    release = create_audited_production_release(bundle, audit, movie, runtime)

    assert release.release_bundle_sha256 == bundle.bundle_sha256
    assert release.movie_sha256 == provenance_for(movie).sha256
    assert release.model_fingerprint == runtime.native_model_manifest_sha256
    assert release.runtime_fingerprint == runtime.fingerprint
    verify_audited_production_release(release, bundle, audit, movie, runtime)


def test_audited_release_rejects_runtime_drift_before_release(tmp_path) -> None:
    runtime = _runtime()
    bundle = _bundle(runtime)
    movie, audit = _audit(tmp_path, runtime)
    drifted_runtime = _runtime(model_digest="e" * 64)

    with pytest.raises(ProductionReleaseError, match="runtime fingerprint"):
        create_audited_production_release(
            bundle,
            audit,
            movie,
            drifted_runtime,
        )


def test_audited_release_rejects_movie_replacement(tmp_path) -> None:
    runtime = _runtime()
    bundle = _bundle(runtime)
    movie, audit = _audit(tmp_path, runtime)
    release = create_audited_production_release(bundle, audit, movie, runtime)

    movie.write_bytes(b"replacement movie bytes")

    with pytest.raises(Exception, match="movie"):
        verify_audited_production_release(release, bundle, audit, movie, runtime)


def test_audited_release_persistence_detects_tampering(tmp_path) -> None:
    runtime = _runtime()
    bundle = _bundle(runtime)
    movie, audit = _audit(tmp_path, runtime)
    release = create_audited_production_release(bundle, audit, movie, runtime)
    release_path = tmp_path / "audited-release.json"
    save_audited_production_release(release, release_path)

    loaded = load_audited_production_release(release_path)
    assert loaded == release

    payload = json.loads(release_path.read_text(encoding="utf-8"))
    payload["release"]["movie_sha256"] = "f" * 64
    release_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ProductionReleaseError, match="integrity hash mismatch"):
        load_audited_production_release(release_path)


def test_audited_release_rejects_malformed_digest() -> None:
    with pytest.raises(ProductionReleaseError, match="movie_sha256"):
        AuditedProductionRelease(
            release_bundle_sha256="a" * 64,
            audit_record_sha256="b" * 64,
            movie_sha256="not-a-digest",
            model_fingerprint="c" * 64,
            runtime_fingerprint="d" * 64,
        )
