import json
from types import SimpleNamespace

from cineos.film.media_probe import probe_media


def _patch_ffprobe(monkeypatch, payload):
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


def test_video_duration_is_authoritative_when_embedded_audio_runs_longer(
    tmp_path, monkeypatch
):
    movie = tmp_path / "shot-with-long-audio.mp4"
    movie.write_bytes(b"encoded-av")
    _patch_ffprobe(
        monkeypatch,
        {
            "format": {"duration": "10.0", "format_name": "mp4"},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "duration": "2.0",
                    "width": 1280,
                    "height": 720,
                    "avg_frame_rate": "24/1",
                    "nb_read_frames": "48",
                },
                {
                    "index": 1,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "duration": "10.0",
                    "sample_rate": "48000",
                    "channels": 2,
                },
            ],
        },
    )

    media = probe_media(movie)

    assert media["duration_seconds"] == 2.0
    assert media["audio_streams"][0]["duration_seconds"] == 10.0


def test_audio_only_media_keeps_container_duration_fallback(tmp_path, monkeypatch):
    audio = tmp_path / "approved-audio.m4a"
    audio.write_bytes(b"encoded-audio")
    _patch_ffprobe(
        monkeypatch,
        {
            "format": {"duration": "10.0", "format_name": "mov,mp4,m4a"},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "duration": "9.98",
                    "sample_rate": "48000",
                    "channels": 2,
                }
            ],
        },
    )

    media = probe_media(audio)

    assert media["duration_seconds"] == 10.0
    assert media["video_stream_count"] == 0
