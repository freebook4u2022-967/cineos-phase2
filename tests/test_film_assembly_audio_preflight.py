from pathlib import Path

import pytest

from cineos.film import assembly
from cineos.film.exceptions import AssemblyError


def _audio_media(*, stream_count=1, duration=10.0):
    streams = [
        {"duration_seconds": duration}
        for _ in range(stream_count)
    ]
    return {
        "audio_stream_count": stream_count,
        "audio_streams": streams,
        "duration_seconds": duration,
    }


def test_preflight_rejects_missing_audio_stream(monkeypatch):
    monkeypatch.setattr(
        assembly,
        "probe_media",
        lambda _path: _audio_media(stream_count=0),
    )

    with pytest.raises(AssemblyError, match="exactly one audio stream"):
        assembly._preflight_audio(Path("approved.wav"), expected_duration=None)


def test_preflight_rejects_multiple_audio_streams(monkeypatch):
    monkeypatch.setattr(
        assembly,
        "probe_media",
        lambda _path: _audio_media(stream_count=2),
    )

    with pytest.raises(AssemblyError, match="exactly one audio stream"):
        assembly._preflight_audio(Path("approved.wav"), expected_duration=None)


def test_preflight_rejects_audio_that_cannot_cover_explicit_timeline(monkeypatch):
    monkeypatch.setattr(
        assembly,
        "probe_media",
        lambda _path: _audio_media(duration=8.0),
    )

    with pytest.raises(AssemblyError, match="cannot cover the requested visual timeline"):
        assembly._preflight_audio(Path("approved.wav"), expected_duration=10.0)


def test_preflight_accepts_audio_within_shortfall_tolerance(monkeypatch):
    monkeypatch.setattr(
        assembly,
        "probe_media",
        lambda _path: _audio_media(duration=9.4),
    )

    evidence = assembly._preflight_audio(
        Path("approved.wav"),
        expected_duration=10.0,
    )

    assert evidence["audio_stream_count"] == 1
    assert evidence["duration_seconds"] == 9.4
    assert evidence["duration_shortfall_seconds"] == pytest.approx(0.6)
