from pathlib import Path
from types import SimpleNamespace

import pytest

from cineos.film.media_probe import MediaProbeError, probe_audio_signal


def test_probe_audio_signal_reports_decoded_volume(tmp_path, monkeypatch):
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"film")
    captured = []

    monkeypatch.setattr(
        "cineos.film.media_probe._ffmpeg", lambda: "/usr/bin/ffmpeg"
    )

    def fake_run(command, **_kwargs):
        captured.extend(command)
        return SimpleNamespace(
            returncode=0,
            stderr=(
                "[Parsed_volumedetect_0] mean_volume: -23.4 dB\n"
                "[Parsed_volumedetect_0] max_volume: -2.1 dB\n"
            ),
        )

    monkeypatch.setattr("cineos.film.media_probe.subprocess.run", fake_run)

    evidence = probe_audio_signal(movie)

    assert evidence == {"mean_volume_db": -23.4, "max_volume_db": -2.1}
    assert "volumedetect" in captured
    assert "0:a:0" in captured


def test_probe_audio_signal_normalizes_negative_infinity_silence(
    tmp_path, monkeypatch
):
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"film")
    monkeypatch.setattr(
        "cineos.film.media_probe._ffmpeg", lambda: "/usr/bin/ffmpeg"
    )
    monkeypatch.setattr(
        "cineos.film.media_probe.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stderr=(
                "mean_volume: -inf dB\n"
                "max_volume: -inf dB\n"
            ),
        ),
    )

    evidence = probe_audio_signal(movie)

    assert evidence == {"mean_volume_db": -120.0, "max_volume_db": -120.0}


def test_probe_audio_signal_fails_closed_on_incomplete_evidence(
    tmp_path, monkeypatch
):
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"film")
    monkeypatch.setattr(
        "cineos.film.media_probe._ffmpeg", lambda: "/usr/bin/ffmpeg"
    )
    monkeypatch.setattr(
        "cineos.film.media_probe.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stderr="mean_volume: -20.0 dB\n",
        ),
    )

    with pytest.raises(MediaProbeError, match="complete audio signal evidence"):
        probe_audio_signal(movie)
