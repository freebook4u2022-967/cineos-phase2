from pathlib import Path
from types import SimpleNamespace

from cineos.film import assembly


def _audio_probe(duration: float) -> dict[str, object]:
    return {
        "audio_stream_count": 1,
        "audio_streams": [
            {
                "codec_name": "pcm_s16le",
                "sample_rate_hz": 48000,
                "channels": 2,
                "duration_seconds": duration,
            }
        ],
        "duration_seconds": duration,
    }


def test_approved_audio_is_explicitly_mapped_over_source_audio(tmp_path, monkeypatch):
    shot = tmp_path / "shot.mp4"
    audio = tmp_path / "approved.wav"
    output = tmp_path / "final.mp4"
    shot.write_bytes(b"shot-with-possible-source-audio")
    audio.write_bytes(b"approved-audio")
    captured: list[str] = []

    monkeypatch.setattr(assembly, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(assembly, "probe_media", lambda _path: _audio_probe(1.0))

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


def test_explicit_durations_hard_trim_each_decoded_shot_before_concat(
    tmp_path, monkeypatch
):
    shots = [tmp_path / "shot-a.mp4", tmp_path / "shot-b.mp4"]
    output = tmp_path / "final.mp4"
    for shot in shots:
        shot.write_bytes(b"shot")
    captured: list[str] = []

    monkeypatch.setattr(assembly, "_ffmpeg", lambda: "ffmpeg")

    def fake_run(command, **_kwargs):
        captured.extend(command)
        Path(command[-1]).write_bytes(b"assembled")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(assembly.subprocess, "run", fake_run)

    assembly.assemble(shots, output, durations=[1.25, 2.5])

    graph = captured[captured.index("-filter_complex") + 1]
    assert (
        "[0:v:0]trim=start=0:duration=1.250000,settb=AVTB,"
        "setpts=PTS-STARTPTS[v0]" in graph
    )
    assert (
        "[1:v:0]trim=start=0:duration=2.500000,settb=AVTB,"
        "setpts=PTS-STARTPTS[v1]" in graph
    )
    assert "[v0][v1]concat=n=2:v=1:a=0[filmv]" in graph
    first_map = captured.index("-map")
    assert captured[first_map : first_map + 2] == ["-map", "[filmv]"]
    assert "-f" not in captured
    assert "concat" not in captured


def test_explicit_durations_map_approved_audio_after_all_shot_inputs(
    tmp_path, monkeypatch
):
    shots = [tmp_path / "shot-a.mp4", tmp_path / "shot-b.mp4"]
    audio = tmp_path / "approved.wav"
    output = tmp_path / "final.mp4"
    for shot in shots:
        shot.write_bytes(b"shot")
    audio.write_bytes(b"approved-audio")
    captured: list[str] = []

    monkeypatch.setattr(assembly, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(assembly, "probe_media", lambda _path: _audio_probe(4.0))

    def fake_run(command, **_kwargs):
        captured.extend(command)
        Path(command[-1]).write_bytes(b"assembled")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(assembly.subprocess, "run", fake_run)

    assembly.assemble(shots, output, durations=[1.0, 2.0], audio_path=audio)

    map_positions = [index for index, value in enumerate(captured) if value == "-map"]
    assert captured[map_positions[0] : map_positions[0] + 2] == ["-map", "[filmv]"]
    assert captured[map_positions[1] : map_positions[1] + 2] == ["-map", "2:a:0"]
    assert "-shortest" not in captured
    assert captured[captured.index("-af") + 1] == "apad"
    assert captured[captured.index("-t") + 1] == "3.000000"
