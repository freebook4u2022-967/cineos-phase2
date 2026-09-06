from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cineos.film.connected_production_evidence import (
    CONNECTED_PRODUCTION_FILM_EVIDENCE_SCHEMA,
    ConnectedProductionFilmEvidenceError,
    connected_production_film_evidence,
    validate_connected_production_film_evidence,
)
from cineos.film.production_assembly import PRODUCTION_EVIDENCE_SCHEMA
from test_atlas_production_connected_evidence import _benchmark


def _canonical_hash(value: dict[str, object]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assembly(tmp_path: Path, benchmark) -> dict[str, object]:
    movie = tmp_path / "final.mp4"
    movie.write_bytes(b"cineos-connected-production-film")
    final_sha = hashlib.sha256(movie.read_bytes()).hexdigest()
    assembly: dict[str, object] = {
        "schema": PRODUCTION_EVIDENCE_SCHEMA,
        "shot_count": len(benchmark.shot_receipts),
        "shots": [
            {
                "index": index,
                "shot_id": receipt.result.shot_id,
                "output_sha256": receipt.output_sha256,
                "evidence_sha256": f"{index + 100:064x}",
            }
            for index, receipt in enumerate(benchmark.shot_receipts)
        ],
        "timeline": {"source": "test-approved-timeline"},
        "audio": None,
        "final_mp4": str(movie),
        "final_mp4_sha256": final_sha,
        "final_media": {"format_name": "mp4"},
    }
    assembly["manifest_sha256"] = _canonical_hash(assembly)
    return assembly


def _resign(assembly: dict[str, object]) -> None:
    assembly.pop("manifest_sha256", None)
    assembly["manifest_sha256"] = _canonical_hash(assembly)


def test_accepts_exact_connected_benchmark_to_final_film_binding(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    assembly = _assembly(tmp_path, benchmark)

    evidence = validate_connected_production_film_evidence(benchmark, assembly)

    assert evidence.accepted is True
    assert evidence.shot_count == 5
    assert evidence.benchmark_chain_sha256 == benchmark.chain_sha256
    assert evidence.final_mp4_sha256 == assembly["final_mp4_sha256"]
    assert evidence.to_dict()["schema"] == CONNECTED_PRODUCTION_FILM_EVIDENCE_SCHEMA
    assert connected_production_film_evidence(benchmark, assembly) is True


def test_rejects_assembly_bound_to_different_connected_artifact(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    assembly = _assembly(tmp_path, benchmark)
    shots = assembly["shots"]
    assert isinstance(shots, list)
    shots[2]["output_sha256"] = f"{999:064x}"
    _resign(assembly)

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="artifact does not match connected benchmark",
    ):
        validate_connected_production_film_evidence(benchmark, assembly)

    assert connected_production_film_evidence(benchmark, assembly) is False


def test_rejects_reordered_connected_shot_timeline(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    assembly = _assembly(tmp_path, benchmark)
    shots = assembly["shots"]
    assert isinstance(shots, list)
    shots[1], shots[2] = shots[2], shots[1]
    _resign(assembly)

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="ID does not match connected benchmark",
    ):
        validate_connected_production_film_evidence(benchmark, assembly)


def test_rejects_tampered_assembly_manifest_without_resigning(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    assembly = _assembly(tmp_path, benchmark)
    assembly["timeline"] = {"source": "substituted-timeline"}

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="manifest SHA-256 does not match its contents",
    ):
        validate_connected_production_film_evidence(benchmark, assembly)


def test_rejects_final_mp4_modified_after_assembly_acceptance(tmp_path) -> None:
    benchmark = _benchmark(tmp_path)
    assembly = _assembly(tmp_path, benchmark)
    movie = Path(str(assembly["final_mp4"]))
    movie.write_bytes(b"substituted-final-film")

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="final MP4 hash does not match assembly evidence",
    ):
        validate_connected_production_film_evidence(benchmark, assembly)

    assert connected_production_film_evidence(benchmark, assembly) is False


def test_rejects_non_hex_qc_evidence_hash_even_when_manifest_is_resigned(
    tmp_path,
) -> None:
    benchmark = _benchmark(tmp_path)
    assembly = _assembly(tmp_path, benchmark)
    shots = assembly["shots"]
    assert isinstance(shots, list)
    shots[1]["evidence_sha256"] = "z" * 64
    _resign(assembly)

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="valid assembly shot 1 evidence SHA-256",
    ):
        validate_connected_production_film_evidence(benchmark, assembly)

    assert connected_production_film_evidence(benchmark, assembly) is False


def test_rejects_reused_qc_evidence_hash_even_when_manifest_is_resigned(
    tmp_path,
) -> None:
    benchmark = _benchmark(tmp_path)
    assembly = _assembly(tmp_path, benchmark)
    shots = assembly["shots"]
    assert isinstance(shots, list)
    shots[3]["evidence_sha256"] = shots[2]["evidence_sha256"]
    _resign(assembly)

    with pytest.raises(
        ConnectedProductionFilmEvidenceError,
        match="reuses QC evidence from another shot",
    ):
        validate_connected_production_film_evidence(benchmark, assembly)

    assert connected_production_film_evidence(benchmark, assembly) is False
