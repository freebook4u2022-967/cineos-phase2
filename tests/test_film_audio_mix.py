from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cineos.film import audio
from cineos.film.audio import AudioTrack


def test_audio_track_rejects_invalid_timeline_values(tmp_path: Path) -> None:
    path = tmp_path / "dialogue.wav"
    with pytest.raises(ValueError, match="gain"):
        AudioTrack(path, gain=-0.1)
    with pytest.raises(ValueError, match="start"):
        AudioTrack(path, start=-1.0)
    with pytest.raises(ValueError, match="fades"):
        AudioTrack(path, fade_out=-0.5)
    with pytest.raises(ValueError, match="kind"):
        AudioTrack(path, kind="  ")


def test_mux_audio_tracks_preserves_video_when_audio_is_missing(tmp_path: Path) -> None:
    video = tmp_path / "picture.mp4"
    video.write_bytes(b"picture")
    output = tmp_path / "final.mp4"

    result = audio.mux_audio_tracks(
        video,
        [AudioTrack(tmp_path / "missing.wav")],
        output,
    )

    assert result == output
    assert output.read_bytes() == b"picture"


def test_mux_audio_tracks_builds_timeline_aware_filter_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "picture.mp4"
    dialogue = tmp_path / "dialogue.wav"
    music = tmp_path / "music.wav"
    output = tmp_path / "final.mp4"
    video.write_bytes(b"video")
    dialogue.write_bytes(b"dialogue")
    music.write_bytes(b"music")
    captured: list[str] = []

    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        captured.extend(command)
        Path(command[-1]).write_bytes(b"mixed")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    result = audio.mux_audio_tracks(
        video,
        [
            AudioTrack(dialogue, kind="dialogue", gain=0.8, fade_in=0.25),
            AudioTrack(
                music,
                kind="music",
                gain=0.4,
                start=1.5,
                fade_out=2.0,
            ),
        ],
        output,
    )

    assert result == output
    assert output.read_bytes() == b"mixed"
    assert captured.count("-i") == 3
    graph = captured[captured.index("-filter_complex") + 1]
    assert "[1:a:0]volume=0.800000,afade=t=in:st=0:d=0.250000[a1]" in graph
    assert "volume=0.400000" in graph
    assert "areverse,afade=t=in:st=0:d=2.000000,areverse" in graph
    assert "adelay=delays=1500:all=1" in graph
    assert "amix=inputs=2:duration=longest:dropout_transition=0" in graph
    assert "alimiter=limit=0.98,apad[aout]" in graph
    assert captured[captured.index("-map") + 1] == "0:v:0"
    assert "[aout]" in captured
    assert "-shortest" in captured


def test_mux_primary_audio_remains_single_track_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "picture.mp4"
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    output = tmp_path / "final.mp4"
    for path in (video, first, second):
        path.write_bytes(path.name.encode())
    captured: list[str] = []

    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        captured.extend(command)
        Path(command[-1]).write_bytes(b"mixed")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    audio.mux_primary_audio(
        video,
        [AudioTrack(first), AudioTrack(second)],
        output,
    )

    assert str(first) in captured
    assert str(second) not in captured
    graph = captured[captured.index("-filter_complex") + 1]
    assert "amix=inputs=1" in graph
