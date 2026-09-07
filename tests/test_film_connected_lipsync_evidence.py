from __future__ import annotations

from pathlib import Path

import pytest

from cineos.audio.lipsync_qc import (
    LipSyncAnalyzerProvenance,
    LipSyncQualityEvidence,
)
from cineos.film.connected_production_evidence import (
    CONNECTED_PRODUCTION_FILM_EVIDENCE_SCHEMA,
    ConnectedProductionFilmEvidenceError,
    connected_production_film_evidence,
    validate_connected_production_film_evidence,
)
from test_atlas_production_connected_evidence import _benchmark
from test_film_connected_production_evidence import _assembly


def _analyzer() -> LipSyncAnalyzerProvenance:
    return LipSyncAnalyzerProvenance(
        analyzer_id="syncnet-production",
        analyzer_revision="4f3c2b1a",
        analyzer_license_id="Apache-2.0",
        analyzer_source_url="https://example.invalid/syncnet-production",
        analyzer_origin="external_pretrained_foundation",
    )


def _report(shot_id: str, video_sha256: str, audio_sha256: str) -> dict[str, object]:
    analyzer = _analyzer()
    return LipSyncQualityEvidence(
        shot_id=shot_id,
        video_sha256=video_sha256,
        audio_sha256=audio_sha256,
        analyzer_origin=analyzer.analyzer_origin,
        analyzer_id=analyzer.analyzer_id,
        analyzer_revision=analyzer.analyzer_revision,
        analyzer_license_id=analyzer.analyzer_license_id,
        analyzer_source_url=analyzer.analyzer_source_url,
        sync_confidence=0.91,
        av_offset_ms=32.0,
        speaking_frame_coverage=0.88,
        face_track_coverage=0.96,
        measured=True,
        accepted=True,
    ).to_dict()


def _dialogue_fixture(tmp_path: Path):
    benchmark = _benchmark(tmp_path)
    assembly = _assembly(tmp_path, benchmark)
    shots = assembly["shots"]
    assert isinstance(shots, list)
    shot = shots[1]
    shot_id = str(shot["shot_id"])
    video_sha = str(shot["output_sha256"])
    audio_sha = f"{12345:064x}"
    report = _report(shot_id, video_sha, audio_sha)
    return benchmark, assembly, shot_id, audio_sha, report


def test_binds_measured_lipsync_to_dialogue_shot_and_final_film(tmp_path) -> None:
    benchmark, assembly, shot_id, audio_sha, report = _dialogue_fixture(tmp_path)

    evidence = validate_connected_production_film_evidence(
        benchmark,
        assembly,
        lipsync_evidence=[report],
        required_dialogue_shot_ids=[shot_id],
        dialogue_audio_sha256_by_shot={shot_id: audio_sha},
        expected_lipsync_analyzer=_analyzer(),
    )

    assert evidence.accepted is True
    assert evidence.dialogue_shot_ids == (shot_id,)
    assert evidence.lipsync_evidence_sha256 == (report["evidence_sha256"],)
    assert evidence.to_dict()["schema"] == CONNECTED_PRODUCTION_FILM_EVIDENCE_SCHEMA
    assert connected_production_film_evidence(
        benchmark,
        assembly,
        lipsync_evidence=[report],
        required_dialogue_shot_ids=[shot_id],
        dialogue_audio_sha256_by_shot={shot_id: audio_sha},
        expected_lipsync_analyzer=_analyzer(),
    ) is True


def test_dialogue_shot_fails_closed_without_lipsync_evidence(tmp_path) -> None:
    benchmark, assembly, shot_id, audio_sha, _ = _dialogue_fixture(tmp_path)

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="requires measured lip-sync evidence",
    ):
        validate_connected_production_film_evidence(
            benchmark,
            assembly,
            required_dialogue_shot_ids=[shot_id],
            dialogue_audio_sha256_by_shot={shot_id: audio_sha},
            expected_lipsync_analyzer=_analyzer(),
        )


def test_rejects_lipsync_report_bound_to_substituted_video(tmp_path) -> None:
    benchmark, assembly, shot_id, audio_sha, _ = _dialogue_fixture(tmp_path)
    substituted = _report(shot_id, f"{99999:064x}", audio_sha)

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="video artifact does not match dialogue shot",
    ):
        validate_connected_production_film_evidence(
            benchmark,
            assembly,
            lipsync_evidence=[substituted],
            required_dialogue_shot_ids=[shot_id],
            dialogue_audio_sha256_by_shot={shot_id: audio_sha},
            expected_lipsync_analyzer=_analyzer(),
        )


def test_rejects_lipsync_report_bound_to_unapproved_dialogue_audio(tmp_path) -> None:
    benchmark, assembly, shot_id, audio_sha, report = _dialogue_fixture(tmp_path)

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="audio artifact does not match dialogue audio",
    ):
        validate_connected_production_film_evidence(
            benchmark,
            assembly,
            lipsync_evidence=[report],
            required_dialogue_shot_ids=[shot_id],
            dialogue_audio_sha256_by_shot={shot_id: f"{54321:064x}"},
            expected_lipsync_analyzer=_analyzer(),
        )


def test_rejects_lipsync_report_from_unpinned_analyzer(tmp_path) -> None:
    benchmark, assembly, shot_id, audio_sha, report = _dialogue_fixture(tmp_path)
    substituted_analyzer = LipSyncAnalyzerProvenance(
        analyzer_id="different-analyzer",
        analyzer_revision="4f3c2b1a",
        analyzer_license_id="Apache-2.0",
        analyzer_source_url="https://example.invalid/syncnet-production",
        analyzer_origin="external_pretrained_foundation",
    )

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="analyzer ID does not match pinned provenance",
    ):
        validate_connected_production_film_evidence(
            benchmark,
            assembly,
            lipsync_evidence=[report],
            required_dialogue_shot_ids=[shot_id],
            dialogue_audio_sha256_by_shot={shot_id: audio_sha},
            expected_lipsync_analyzer=substituted_analyzer,
        )


def test_rejects_incomplete_lipsync_coverage_for_required_dialogue_shots(tmp_path) -> None:
    benchmark, assembly, shot_id, audio_sha, report = _dialogue_fixture(tmp_path)
    shots = assembly["shots"]
    assert isinstance(shots, list)
    second_shot_id = str(shots[2]["shot_id"])

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="missing measured lip-sync evidence",
    ):
        validate_connected_production_film_evidence(
            benchmark,
            assembly,
            lipsync_evidence=[report],
            required_dialogue_shot_ids=[shot_id, second_shot_id],
            dialogue_audio_sha256_by_shot={
                shot_id: audio_sha,
                second_shot_id: f"{67890:064x}",
            },
            expected_lipsync_analyzer=_analyzer(),
        )


def test_non_dialogue_legacy_connected_film_remains_compatible(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    assembly = _assembly(tmp_path, benchmark)

    evidence = validate_connected_production_film_evidence(benchmark, assembly)

    assert evidence.accepted is True
    assert evidence.dialogue_shot_ids == ()
    assert evidence.lipsync_evidence_sha256 == ()
