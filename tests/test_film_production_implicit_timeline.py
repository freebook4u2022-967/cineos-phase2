import hashlib
from pathlib import Path

import pytest

from cineos.film.exceptions import AssemblyError
from cineos.film.production_assembly import assemble_production_film


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(tmp_path: Path, count: int = 5):
    records = []
    for index in range(count):
        path = tmp_path / f"shot-{index}.mp4"
        path.write_bytes(f"shot-{index}".encode())
        records.append(
            {
                "shot_id": f"shot-{index}",
                "accepted": True,
                "decision": "accept",
                "production_gpu_evidence": True,
                "output_path": str(path),
                "output_sha256": _sha(path),
                "evidence_sha256": hashlib.sha256(
                    f"evidence-{index}".encode()
                ).hexdigest(),
            }
        )
    return records


def _probe(duration: float) -> dict:
    return {
        "duration_seconds": duration,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "video_stream_count": 1,
        "audio_stream_count": 0,
        "video_codecs": ["h264"],
        "audio_codecs": [],
        "video_dimensions": [{"width": 1280, "height": 720}],
        "audio_streams": [],
    }


def test_implicit_timeline_uses_probed_shot_durations(tmp_path, monkeypatch):
    records = _records(tmp_path)
    output = tmp_path / "final.mp4"

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"assembled-film")
        return Path(destination)

    def fake_probe(path):
        return _probe(5.0 if Path(path).name == "final.mp4" else 1.0)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr("cineos.film.production_assembly.probe_media", fake_probe)

    manifest = assemble_production_film(records, output)

    assert manifest["schema"] == "cineos-production-film-evidence/0.8"
    assert manifest["timeline"]["source"] == "probed-source-shots"
    assert manifest["timeline"]["expected_duration_seconds"] == 5.0
    assert manifest["timeline"]["compatibility"] == {
        "width": 1280,
        "height": 720,
        "frame_rate": None,
        "edit_durations_seconds": None,
    }


def test_implicit_timeline_rejects_truncated_final_encode(tmp_path, monkeypatch):
    records = _records(tmp_path)

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"truncated-film")
        return Path(destination)

    def fake_probe(path):
        return _probe(3.0 if Path(path).name == "final.mp4" else 1.0)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr("cineos.film.production_assembly.probe_media", fake_probe)

    with pytest.raises(
        AssemblyError, match="duration deviates from the approved visual timeline"
    ):
        assemble_production_film(records, tmp_path / "final.mp4")


def test_explicit_edit_durations_remain_authoritative(tmp_path, monkeypatch):
    records = _records(tmp_path)
    output = tmp_path / "final.mp4"

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"edited-film")
        return Path(destination)

    def fake_probe(path):
        return _probe(5.0 if Path(path).name == "final.mp4" else 2.0)

    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)
    monkeypatch.setattr("cineos.film.production_assembly.probe_media", fake_probe)

    manifest = assemble_production_film(
        records,
        output,
        durations=[1.0] * 5,
    )

    assert manifest["timeline"]["source"] == "explicit-edit-durations"
    assert manifest["timeline"]["expected_duration_seconds"] == 5.0
    assert manifest["timeline"]["compatibility"]["edit_durations_seconds"] == [1.0] * 5
