from pathlib import Path

import pytest

from cineos.film.assembly import assemble
from cineos.film.exceptions import AssemblyError


def test_rejects_output_that_overwrites_source_video(tmp_path: Path, monkeypatch):
    shot = tmp_path / "shot.mp4"
    shot.write_bytes(b"approved-shot")
    monkeypatch.setattr(
        "cineos.film.assembly._ffmpeg",
        lambda: pytest.fail("FFmpeg must not run when output aliases an input"),
    )

    with pytest.raises(AssemblyError, match="output must be distinct"):
        assemble([shot], shot)

    assert shot.read_bytes() == b"approved-shot"


def test_rejects_output_that_overwrites_approved_audio(tmp_path: Path, monkeypatch):
    shot = tmp_path / "shot.mp4"
    audio = tmp_path / "mix.wav"
    shot.write_bytes(b"approved-shot")
    audio.write_bytes(b"approved-audio")
    monkeypatch.setattr(
        "cineos.film.assembly._ffmpeg",
        lambda: pytest.fail("FFmpeg must not run when output aliases approved audio"),
    )

    with pytest.raises(AssemblyError, match="output must be distinct"):
        assemble([shot], audio, audio_path=audio)

    assert shot.read_bytes() == b"approved-shot"
    assert audio.read_bytes() == b"approved-audio"
