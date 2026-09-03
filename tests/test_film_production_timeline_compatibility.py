import hashlib
from pathlib import Path

import pytest

from cineos.film.exceptions import AssemblyError
from cineos.film.production_assembly import assemble_production_film


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(tmp_path: Path):
    records = []
    for index in range(5):
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


def _media(*, width=1280, height=720, duration=2.0):
    return {
        "duration_seconds": duration,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "video_stream_count": 1,
        "audio_stream_count": 0,
        "video_codecs": ["h264"],
        "audio_codecs": [],
        "video_dimensions": [{"width": width, "height": height}],
        "audio_streams": [],
    }


def test_rejects_mixed_frame_geometry_before_ffmpeg(tmp_path, monkeypatch):
    records = _records(tmp_path)

    def fake_probe(path):
        index = int(Path(path).stem.split("-")[-1])
        if index == 3:
            return _media(width=1920, height=1080)
        return _media()

    monkeypatch.setattr("cineos.film.production_assembly.probe_media", fake_probe)
    monkeypatch.setattr(
        "cineos.film.production_assembly.assemble",
        lambda *_args, **_kwargs: pytest.fail("FFmpeg must not run"),
    )

    with pytest.raises(AssemblyError, match="identical frame dimensions"):
        assemble_production_film(records, tmp_path / "final.mp4")


def test_rejects_edit_duration_longer_than_approved_source(tmp_path, monkeypatch):
    records = _records(tmp_path)
    monkeypatch.setattr(
        "cineos.film.production_assembly.probe_media",
        lambda _path: _media(duration=2.0),
    )
    monkeypatch.setattr(
        "cineos.film.production_assembly.assemble",
        lambda *_args, **_kwargs: pytest.fail("FFmpeg must not run"),
    )

    with pytest.raises(AssemblyError, match="exceeds approved source duration"):
        assemble_production_film(
            records,
            tmp_path / "final.mp4",
            durations=[1.0, 1.0, 2.2, 1.0, 1.0],
        )


def test_allows_small_probe_rounding_tolerance(tmp_path, monkeypatch):
    records = _records(tmp_path)
    output = tmp_path / "final.mp4"

    def fake_probe(path):
        path = Path(path)
        if path.name == "final.mp4":
            return _media(duration=10.10)
        return _media(duration=2.02)

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.probe_media", fake_probe)
    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)

    manifest = assemble_production_film(
        records,
        output,
        durations=[2.02] * 5,
    )

    compatibility = manifest["timeline"]["compatibility"]
    assert compatibility["width"] == 1280
    assert compatibility["height"] == 720
    assert compatibility["edit_durations_seconds"] == [2.02] * 5
