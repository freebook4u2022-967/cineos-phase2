from pathlib import Path
from types import SimpleNamespace

from cineos.film import assembly


def test_approved_audio_is_explicitly_mapped_over_source_audio(tmp_path, monkeypatch):
    shot = tmp_path / "shot.mp4"
    audio = tmp_path / "approved.wav"
    output = tmp_path / "final.mp4"
    shot.write_bytes(b"shot-with-possible-source-audio")
    audio.write_bytes(b"approved-audio")
    captured: list[str] = []

    monkeypatch.setattr(assembly, "_ffmpeg", lambda: "ffmpeg")

    def fake_run(command, **_kwargs):
        captured.extend(command)
        Path(command[-1]).write_bytes(b"assembled")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(assembly.subprocess, "run", fake_run)

    result = assembly.assemble([shot], output, audio_path=audio)

    assert result == output
    first_map = captured.index("-map")
    assert captured[first_map : first_map + 4] == [
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ]
    assert "-an" not in captured


def test_video_only_assembly_strips_any_source_audio(tmp_path, monkeypatch):
    shot = tmp_path / "shot.mp4"
    output = tmp_path / "final.mp4"
    shot.write_bytes(b"shot-with-possible-source-audio")
    captured: list[str] = []

    monkeypatch.setattr(assembly, "_ffmpeg", lambda: "ffmpeg")

    def fake_run(command, **_kwargs):
        captured.extend(command)
        Path(command[-1]).write_bytes(b"assembled")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(assembly.subprocess, "run", fake_run)

    assembly.assemble([shot], output)

    assert "-an" in captured
    assert "-map" not in captured
