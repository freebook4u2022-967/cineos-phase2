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


def _media(**overrides):
    media = {
        "duration_seconds": 1.0,
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "video_stream_count": 1,
        "audio_stream_count": 0,
        "video_codecs": ["h264"],
        "audio_codecs": [],
        "video_dimensions": [{"width": 1280, "height": 720}],
        "audio_streams": [],
    }
    media.update(overrides)
    return media


def _fail_assembly(*_args, **_kwargs):
    pytest.fail("assembly must not run for invalid bound shot media")


@pytest.mark.parametrize(
    ("media", "message"),
    [
        ({**_media(), "format_name": "matroska,webm"}, "not an MP4 container"),
        ({**_media(), "video_stream_count": 0}, "exactly one video stream"),
        ({**_media(), "video_codecs": ["hevc"]}, "exactly one H.264"),
        (
            {**_media(), "video_dimensions": [{"width": 1279, "height": 720}]},
            "invalid H.264/yuv420p",
        ),
        ({**_media(), "duration_seconds": 0.0}, "no finite positive duration"),
    ],
)
def test_rejects_invalid_bound_shot_media_before_assembly(
    tmp_path, monkeypatch, media, message
):
    records = _records(tmp_path)
    bad_path = Path(records[2]["output_path"]).resolve()

    def fake_probe(path):
        if Path(path).resolve() == bad_path:
            return media
        return _media()

    monkeypatch.setattr("cineos.film.production_assembly.probe_media", fake_probe)
    monkeypatch.setattr("cineos.film.production_assembly.assemble", _fail_assembly)

    with pytest.raises(AssemblyError, match=message):
        assemble_production_film(records, tmp_path / "final.mp4")


def test_records_independently_probed_shot_media_in_manifest(tmp_path, monkeypatch):
    records = _records(tmp_path)

    def fake_probe(path):
        path = Path(path)
        if path.name == "final.mp4":
            return _media(duration_seconds=6.0)
        index = int(path.stem.split("-")[-1])
        audio_count = 1 if index == 2 else 0
        return _media(
            duration_seconds=1.0 + index / 10,
            audio_stream_count=audio_count,
        )

    def fake_assemble(_shots, destination, **_kwargs):
        Path(destination).write_bytes(b"assembled-film")
        return Path(destination)

    monkeypatch.setattr("cineos.film.production_assembly.probe_media", fake_probe)
    monkeypatch.setattr("cineos.film.production_assembly.assemble", fake_assemble)

    manifest = assemble_production_film(records, tmp_path / "final.mp4")

    assert manifest["shots"][0]["media"]["video_codec"] == "h264"
    assert manifest["shots"][2]["media"]["audio_stream_count"] == 1
    assert manifest["shots"][4]["media"]["duration_seconds"] == 1.4
    assert manifest["timeline"]["expected_duration_seconds"] == 6.0
