import json
from types import SimpleNamespace

import pytest

from cineos.film.media_probe import MediaProbeError, probe_media


def _run_probe(tmp_path, monkeypatch, payload):
    movie = tmp_path / "shot.mp4"
    movie.write_bytes(b"encoded-video")
    monkeypatch.setattr(
        "cineos.film.media_probe.shutil.which",
        lambda executable: f"/usr/bin/{executable}",
    )
    monkeypatch.setattr(
        "cineos.film.media_probe.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )
    return probe_media(movie)


def _video_stream(**overrides):
    stream = {
        "index": 0,
        "codec_type": "video",
        "codec_name": "h264",
        "duration": "2.0",
        "width": 1280,
        "height": 720,
        "avg_frame_rate": "24/1",
        "nb_read_frames": "48",
    }
    stream.update(overrides)
    return stream


def test_probe_media_rejects_malformed_explicit_stream_duration(tmp_path, monkeypatch):
    payload = {
        "format": {"duration": "2.0", "format_name": "mp4"},
        "streams": [_video_stream(duration="not-a-duration")],
    }

    with pytest.raises(MediaProbeError, match="malformed explicit duration evidence"):
        _run_probe(tmp_path, monkeypatch, payload)


def test_probe_media_rejects_malformed_explicit_container_duration(tmp_path, monkeypatch):
    payload = {
        "format": {"duration": "corrupt", "format_name": "mp4"},
        "streams": [_video_stream()],
    }

    with pytest.raises(MediaProbeError, match="malformed explicit duration evidence"):
        _run_probe(tmp_path, monkeypatch, payload)


def test_probe_media_allows_na_duration_when_decoded_timing_is_valid(tmp_path, monkeypatch):
    payload = {
        "format": {"duration": "N/A", "format_name": "mp4"},
        "streams": [_video_stream(duration="N/A")],
    }

    media = _run_probe(tmp_path, monkeypatch, payload)

    assert media["duration_seconds"] == pytest.approx(2.0)
    assert media["video_frame_counts"] == [48]


def test_probe_media_rejects_non_object_stream_metadata(tmp_path, monkeypatch):
    payload = {
        "format": {"duration": "2.0", "format_name": "mp4"},
        "streams": [_video_stream(), "corrupt-stream"],
    }

    with pytest.raises(MediaProbeError, match="malformed stream metadata"):
        _run_probe(tmp_path, monkeypatch, payload)


def test_probe_media_rejects_non_object_format_metadata(tmp_path, monkeypatch):
    payload = {
        "format": "corrupt-format",
        "streams": [_video_stream()],
    }

    with pytest.raises(MediaProbeError, match="malformed format metadata"):
        _run_probe(tmp_path, monkeypatch, payload)


def test_probe_media_rejects_non_object_top_level_json(tmp_path, monkeypatch):
    with pytest.raises(MediaProbeError, match="must be a JSON object"):
        _run_probe(tmp_path, monkeypatch, ["not", "an", "object"])
