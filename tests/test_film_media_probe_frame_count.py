import json
from types import SimpleNamespace

from cineos.film.media_probe import probe_media


def _patch_ffprobe(monkeypatch, payload, captured):
    monkeypatch.setattr(
        "cineos.film.media_probe.shutil.which",
        lambda executable: f"/usr/bin/{executable}",
    )

    def fake_run(command, **_kwargs):
        captured.extend(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr("cineos.film.media_probe.subprocess.run", fake_run)


def test_probe_media_decodes_and_counts_video_frames(tmp_path, monkeypatch):
    movie = tmp_path / "shot.mp4"
    movie.write_bytes(b"encoded-video")
    captured = []
    _patch_ffprobe(
        monkeypatch,
        {
            "format": {"duration": "2.0", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
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
                }
            ],
        },
        captured,
    )

    media = probe_media(movie)

    assert "-count_frames" in captured
    assert "nb_read_frames" in captured[captured.index("-show_entries") + 1]
    assert media["video_frame_counts"] == [48]
    assert media["video_frame_rates"] == ["24/1"]


def test_probe_media_marks_unavailable_frame_count_as_unverified(tmp_path, monkeypatch):
    movie = tmp_path / "shot.mp4"
    movie.write_bytes(b"encoded-video")
    captured = []
    _patch_ffprobe(
        monkeypatch,
        {
            "format": {"duration": "2.0", "format_name": "mp4"},
            "streams": [
                {
                    "index": 0,
                    "codec_type": "video",
                    "codec_name": "h264",
                    "duration": "2.0",
                    "width": 1280,
                    "height": 720,
                    "avg_frame_rate": "24/1",
                    "nb_read_frames": "N/A",
                }
            ],
        },
        captured,
    )

    media = probe_media(movie)

    assert media["video_frame_counts"] == [None]
