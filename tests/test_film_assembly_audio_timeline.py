from __future__ import annotations

from types import SimpleNamespace

import pytest

from cineos.film import assembly as assembly_module
from cineos.film.exceptions import AssemblyError


def _audio_probe(duration: float) -> dict[str, object]:
    return {
        "audio_stream_count": 1,
        "audio_streams": [{"duration_seconds": duration}],
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
