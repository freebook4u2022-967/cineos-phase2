from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.film import assembly as assembly_module
from cineos.film.exceptions import AssemblyError


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


def _capture_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    captured: list[str] = []

    monkeypatch.setattr(assembly_module, "file_hash", lambda _path: "0" * 64)
    monkeypatch.setattr(assembly_module, "_ffmpeg", lambda: "ffmpeg")

    def fake_run(command, **_kwargs):
        captured.extend(command)
        output = assembly_module.Path(command[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mp4")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(assembly_module.subprocess, "run", fake_run)
    return captured


def test_short_approved_audio_is_padded_without_shortening_explicit_video(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured = _capture_ffmpeg(monkeypatch)
    monkeypatch.setattr(
        assembly_module,
        "probe_media",
        lambda _path: _audio_probe(59.3),
    )

    output = assembly_module.assemble(
        [tmp_path / "shot-a.mp4", tmp_path / "shot-b.mp4"],
        tmp_path / "film.mp4",
        durations=[30.0, 30.0],
        audio_path=tmp_path / "mix.wav",
    )

    assert output == tmp_path / "film.mp4"
    assert "-shortest" not in captured
    assert captured[captured.index("-af") + 1] == "apad"
    assert captured[captured.index("-t") + 1] == "60.000000"
    assert captured[captured.index("-map") + 1] == "[filmv]"
    audio_map_index = captured.index("-map", captured.index("-map") + 1)
    assert captured[audio_map_index + 1] == "2:a:0"


def test_untrimmed_audio_mux_derives_visual_duration_from_shots(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured = _capture_ffmpeg(monkeypatch)

    def fake_probe(path):
        if path.name == "mix.wav":
            return _audio_probe(59.4)
        return {"duration_seconds": 30.0}

    monkeypatch.setattr(assembly_module, "probe_media", fake_probe)

    assembly_module.assemble(
        [tmp_path / "shot-a.mp4", tmp_path / "shot-b.mp4"],
        tmp_path / "film.mp4",
        audio_path=tmp_path / "mix.wav",
    )

    assert "-shortest" not in captured
    assert captured[captured.index("-af") + 1] == "apad"
    assert captured[captured.index("-t") + 1] == "60.000000"
    first_map = captured.index("-map")
    assert captured[first_map + 1] == "0:v:0"
    second_map = captured.index("-map", first_map + 1)
    assert captured[second_map + 1] == "1:a:0"


def test_non_finite_edit_duration_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(assembly_module, "file_hash", lambda _path: "0" * 64)

    with pytest.raises(AssemblyError, match="finite and positive"):
        assembly_module.assemble(
            [tmp_path / "shot.mp4"],
            tmp_path / "film.mp4",
            durations=[float("nan")],
        )


def test_non_finite_approved_audio_duration_fails_closed_before_encode(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(assembly_module, "file_hash", lambda _path: "0" * 64)
    monkeypatch.setattr(
        assembly_module,
        "probe_media",
        lambda _path: _audio_probe(float("nan")),
    )

    def fail_if_encoded(*_args, **_kwargs):
        raise AssertionError("FFmpeg must not run for invalid approved audio evidence")

    monkeypatch.setattr(assembly_module.subprocess, "run", fail_if_encoded)

    with pytest.raises(AssemblyError, match="finite positive duration"):
        assembly_module.assemble(
            [tmp_path / "shot.mp4"],
            tmp_path / "film.mp4",
            durations=[5.0],
            audio_path=tmp_path / "mix.wav",
        )


def test_crossfade_builds_decoded_frame_transitions_and_shortens_audio_timeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    captured = _capture_ffmpeg(monkeypatch)
    monkeypatch.setattr(
        assembly_module,
        "probe_media",
        lambda _path: _audio_probe(14.0),
    )

    assembly_module.assemble(
        [
            tmp_path / "shot-a.mp4",
            tmp_path / "shot-b.mp4",
            tmp_path / "shot-c.mp4",
        ],
        tmp_path / "film.mp4",
        durations=[5.0, 5.0, 5.0],
        crossfade=0.5,
        audio_path=tmp_path / "mix.wav",
    )

    graph = captured[captured.index("-filter_complex") + 1]
    assert "xfade=transition=fade:duration=0.500000:offset=4.500000[xf1]" in graph
    assert (
        "[xf1][v2]xfade=transition=fade:duration=0.500000:offset=9.000000[filmv]"
        in graph
    )
    assert captured[captured.index("-t") + 1] == "14.000000"
    assert captured[captured.index("-map") + 1] == "[filmv]"


def test_crossfade_requires_explicit_durations(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(assembly_module, "file_hash", lambda _path: "0" * 64)

    with pytest.raises(AssemblyError, match="requires explicit shot durations"):
        assembly_module.assemble(
            [tmp_path / "shot-a.mp4", tmp_path / "shot-b.mp4"],
            tmp_path / "film.mp4",
            crossfade=0.5,
        )


def test_crossfade_must_be_finite_and_shorter_than_every_edit(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(assembly_module, "file_hash", lambda _path: "0" * 64)

    with pytest.raises(AssemblyError, match="finite and non-negative"):
        assembly_module.assemble(
            [tmp_path / "shot-a.mp4", tmp_path / "shot-b.mp4"],
            tmp_path / "film.mp4",
            durations=[5.0, 5.0],
            crossfade=float("nan"),
        )

    with pytest.raises(AssemblyError, match="shorter than every shot duration"):
        assembly_module.assemble(
            [tmp_path / "shot-a.mp4", tmp_path / "shot-b.mp4"],
            tmp_path / "film.mp4",
            durations=[0.5, 5.0],
            crossfade=0.5,
        )
