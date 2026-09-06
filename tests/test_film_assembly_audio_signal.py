from __future__ import annotations

from pathlib import Path

import pytest

from cineos.film import assembly
from cineos.film.exceptions import AssemblyError
from cineos.film.media_probe import MediaProbeError


def _valid_media(*, audio_stream_count: int) -> dict[str, object]:
    return {
        "duration_seconds": 10.0,
        "video_stream_count": 1,
        "audio_stream_count": audio_stream_count,
        "video_frame_counts": [240],
    }


def test_postflight_audio_requires_measurable_decoded_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = Path("film.mp4")
    monkeypatch.setattr(
        assembly,
        "probe_media",
        lambda _path: _valid_media(audio_stream_count=1),
    )
    monkeypatch.setattr(
        assembly,
        "probe_audio_signal",
        lambda _path: {"mean_volume_db": -24.0, "max_volume_db": -3.0},
    )

    media = assembly._postflight_output(
        destination,
        expected_duration=10.0,
        expect_audio=True,
    )

    assert media["audio_stream_count"] == 1


def test_postflight_rejects_digitally_silent_soundtrack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = Path("film.mp4")
    monkeypatch.setattr(
        assembly,
        "probe_media",
        lambda _path: _valid_media(audio_stream_count=1),
    )
    monkeypatch.setattr(
        assembly,
        "probe_audio_signal",
        lambda _path: {"mean_volume_db": -120.0, "max_volume_db": -120.0},
    )

    with pytest.raises(AssemblyError, match="contains no measurable signal"):
        assembly._postflight_output(
            destination,
            expected_duration=10.0,
            expect_audio=True,
        )


def test_postflight_rejects_non_finite_audio_signal_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = Path("film.mp4")
    monkeypatch.setattr(
        assembly,
        "probe_media",
        lambda _path: _valid_media(audio_stream_count=1),
    )
    monkeypatch.setattr(
        assembly,
        "probe_audio_signal",
        lambda _path: {"mean_volume_db": float("nan"), "max_volume_db": -2.0},
    )

    with pytest.raises(AssemblyError, match="non-finite audio-signal evidence"):
        assembly._postflight_output(
            destination,
            expected_duration=10.0,
            expect_audio=True,
        )


def test_postflight_fails_closed_when_audio_signal_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = Path("film.mp4")
    monkeypatch.setattr(
        assembly,
        "probe_media",
        lambda _path: _valid_media(audio_stream_count=1),
    )

    def fail_probe(_path: Path) -> dict[str, float]:
        raise MediaProbeError("decode failed")

    monkeypatch.setattr(assembly, "probe_audio_signal", fail_probe)

    with pytest.raises(AssemblyError, match="audio-signal postflight failed"):
        assembly._postflight_output(
            destination,
            expected_duration=10.0,
            expect_audio=True,
        )


def test_video_only_postflight_does_not_probe_audio_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = Path("film.mp4")
    monkeypatch.setattr(
        assembly,
        "probe_media",
        lambda _path: _valid_media(audio_stream_count=0),
    )

    def unexpected_probe(_path: Path) -> dict[str, float]:
        raise AssertionError(
            "video-only output must not invoke audio signal inspection"
        )

    monkeypatch.setattr(assembly, "probe_audio_signal", unexpected_probe)

    media = assembly._postflight_output(
        destination,
        expected_duration=10.0,
        expect_audio=False,
    )

    assert media["audio_stream_count"] == 0
