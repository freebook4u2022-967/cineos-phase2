import json
from dataclasses import dataclass

import pytest

from cineos.native_video.release_receipt import (
    ProductionReleaseError,
    create_production_film_receipt,
    load_production_film_receipt,
    save_production_film_receipt,
    verify_production_film_receipt,
)
from cineos.native_video.runtime_manifest import (
    LEGACY_UNBOUND_NATIVE_MODEL_MANIFEST,
    ProductionRuntimeManifest,
)


TEST_MODEL_MANIFEST_SHA256 = "a" * 64


@dataclass(frozen=True)
class _FinalQC:
    decision: str
    directives: tuple[str, ...] = ()

    def as_dict(self):
        return {
            "decision": self.decision,
            "directives": list(self.directives),
        }


def _manifest(*, model_manifest: str = TEST_MODEL_MANIFEST_SHA256):
    return ProductionRuntimeManifest(
        renderer_id="cineos-native",
        temporal_model_fingerprint="temporal-weights-sha",
        device="cpu",
        max_recovery_attempts=2,
        require_final_film_evaluation=True,
        require_audio=True,
        final_gate_policy_fingerprint="final-gate-policy-sha",
        native_model_manifest_sha256=model_manifest,
    )


def _plan():
    return [
        {
            "shot_id": "shot-001",
            "scene_id": "scene-001",
            "duration": 4.0,
            "character_ids": ["lead"],
        },
        {
            "shot_id": "shot-002",
            "scene_id": "scene-001",
            "duration": 4.0,
            "character_ids": ["lead"],
        },
    ]


def test_production_receipt_binds_artifact_plan_runtime_qc_and_build(tmp_path):
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-final-movie")
    qc = _FinalQC("accept")
    manifest = _manifest()

    receipt = create_production_film_receipt(
        movie,
        _plan(),
        manifest,
        qc,
        build_content_hash="film-build-content-sha",
    )

    assert receipt.artifact_size_bytes == len(b"cineos-final-movie")
    assert receipt.final_qc_decision == "accept"
    assert receipt.renderer_id == "cineos-native"
    assert receipt.native_model_manifest_sha256 == TEST_MODEL_MANIFEST_SHA256
    assert len(receipt.receipt_sha256) == 64
    verify_production_film_receipt(
        receipt,
        movie,
        _plan(),
        manifest,
        qc,
        build_content_hash="film-build-content-sha",
    )


def test_production_receipt_refuses_rejected_final_qc(tmp_path):
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-final-movie")

    with pytest.raises(ProductionReleaseError, match="rejected final QC"):
        create_production_film_receipt(
            movie,
            _plan(),
            _manifest(),
            _FinalQC("reject", ("temporal drift",)),
            build_content_hash="film-build-content-sha",
        )


def test_production_receipt_requires_released_native_model_by_default(tmp_path):
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-final-movie")
    manifest = _manifest(model_manifest=LEGACY_UNBOUND_NATIVE_MODEL_MANIFEST)

    with pytest.raises(ProductionReleaseError, match="bound native model manifest"):
        create_production_film_receipt(
            movie,
            _plan(),
            manifest,
            _FinalQC("accept"),
            build_content_hash="film-build-content-sha",
        )

    legacy = create_production_film_receipt(
        movie,
        _plan(),
        manifest,
        _FinalQC("accept"),
        build_content_hash="film-build-content-sha",
        require_released_model=False,
    )
    assert legacy.native_model_manifest_sha256 == LEGACY_UNBOUND_NATIVE_MODEL_MANIFEST


def test_receipt_verification_detects_artifact_tampering(tmp_path):
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-final-movie")
    manifest = _manifest()
    qc = _FinalQC("warn", ("minor boundary variance",))
    receipt = create_production_film_receipt(
        movie,
        _plan(),
        manifest,
        qc,
        build_content_hash="film-build-content-sha",
    )

    movie.write_bytes(b"tampered-final-movie")
    with pytest.raises(ProductionReleaseError, match="artifact_sha256"):
        verify_production_film_receipt(
            receipt,
            movie,
            _plan(),
            manifest,
            qc,
            build_content_hash="film-build-content-sha",
        )


def test_receipt_verification_detects_plan_or_qc_drift(tmp_path):
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-final-movie")
    manifest = _manifest()
    qc = _FinalQC("accept")
    receipt = create_production_film_receipt(
        movie,
        _plan(),
        manifest,
        qc,
        build_content_hash="film-build-content-sha",
    )

    changed_plan = _plan()
    changed_plan[1]["duration"] = 5.0
    with pytest.raises(ProductionReleaseError, match="plan_sha256"):
        verify_production_film_receipt(
            receipt,
            movie,
            changed_plan,
            manifest,
            qc,
            build_content_hash="film-build-content-sha",
        )

    with pytest.raises(ProductionReleaseError, match="final_qc_sha256"):
        verify_production_film_receipt(
            receipt,
            movie,
            _plan(),
            manifest,
            _FinalQC("accept", ("new evidence",)),
            build_content_hash="film-build-content-sha",
        )


def test_persisted_receipt_has_independent_integrity_hash(tmp_path):
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"cineos-final-movie")
    receipt = create_production_film_receipt(
        movie,
        _plan(),
        _manifest(),
        _FinalQC("accept"),
        build_content_hash="film-build-content-sha",
    )
    path = save_production_film_receipt(receipt, tmp_path / "release.json")
    restored = load_production_film_receipt(path)
    assert restored == receipt

    document = json.loads(path.read_text(encoding="utf-8"))
    document["receipt"]["build_content_hash"] = "tampered-build"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ProductionReleaseError, match="integrity hash mismatch"):
        load_production_film_receipt(path)
