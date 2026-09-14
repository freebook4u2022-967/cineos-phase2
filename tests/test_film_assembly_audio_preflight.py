from __future__ import annotations

from pathlib import Path

import pytest

from cineos.film.assembly import _preflight_audio
from cineos.film.exceptions import AssemblyError


def _media(**stream_overrides):
    stream = {
        "codec_name": "pcm_s16le",
        "sample_rate_hz": 48_000,
        "channels": 2,
        "duration_seconds": 10.0,
    }
    stream.update(stream_overrides)
    return {
        "audio_stream_count": 1,
        "audio_streams": [stream],
        "duration_seconds": 10.0,
    }


def test_audio_preflight_records_decodable_stream_evidence(monkeypatch):
    monkeypatch.setattr("cineos.film.assembly.probe_media", lambda _: _media())

    evidence = _preflight_audio(Path("approved.wav"), expected_duration=9.5)

    assert evidence["codec_name"] == "pcm_s16le"
    assert evidence["sample_rate_hz"] == 48_000
    assert evidence["channels"] == 2
    assert evidence["duration_seconds"] == 10.0
    assert evidence["duration_shortfall_seconds"] == pytest.approx(-0.5)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"codec_name": ""}, "missing codec evidence"),
        ({"sample_rate_hz": None}, "no valid sample-rate evidence"),
        ({"sample_rate_hz": 0}, "no valid sample-rate evidence"),
        ({"channels": None}, "no valid channel-count evidence"),
        ({"channels": 0}, "no valid channel-count evidence"),
    ],
)
def test_audio_preflight_rejects_incomplete_decode_evidence(
    monkeypatch, overrides, message
):
    monkeypatch.setattr(
        "cineos.film.assembly.probe_media",
        lambda _: _media(**overrides),
    )

    with pytest.raises(AssemblyError, match=message):
        _preflight_audio(Path("approved.wav"), expected_duration=9.5)


def test_audio_preflight_rejects_nonfinite_visual_duration(monkeypatch):
    monkeypatch.setattr("cineos.film.assembly.probe_media", lambda _: _media())

    with pytest.raises(
        AssemblyError,
        match="approved visual timeline has no finite positive duration",
    ):
        _preflight_audio(Path("approved.wav"), expected_duration=float("nan"))
