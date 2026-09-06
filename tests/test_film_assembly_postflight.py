from pathlib import Path

import pytest

from cineos.film import assembly
from cineos.film.exceptions import AssemblyError


def _media(
    *,
    duration: float = 10.0,
    video: int = 1,
    audio: int = 0,
    frame_counts: list[int | None] | None = None,
) -> dict:
    return {
        "duration_seconds": duration,
        "video_stream_count": video,
        "audio_stream_count": audio,
        "video_frame_counts": [240] if frame_counts is None else frame_counts,
        "video_codecs": ["h264"] * video,
        "video_dimensions": [{"width": 1920, "height": 1080}] * video,
        "video_frame_rates": ["24/1"] * video,
        "audio_streams": (
            [
                {
                    "codec_name": "aac",
                    "sample_rate_hz": 48_000,
                    "channels": 2,
                    "duration_seconds": duration,
                }
            ]
            if audio
            else []
        ),
    }


def test_postflight_accepts_video_only_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assembly, "probe_media", lambda _path: _media())

    evidence = assembly._postflight_output(
        Path("film.mp4"), expected_duration=10.0, expect_audio=False
    )

    assert evidence["video_stream_count"] == 1
    assert evidence["audio_stream_count"] == 0
    assert evidence["video_frame_counts"] == [240]


def test_postflight_rejects_truncated_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        assembly,
        "probe_media",
        lambda _path: _media(duration=8.0),
    )

    with pytest.raises(AssemblyError, match="timeline drift exceeds tolerance"):
        assembly._postflight_output(
            Path("film.mp4"), expected_duration=10.0, expect_audio=False
        )


def test_postflight_rejects_missing_approved_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(assembly, "probe_media", lambda _path: _media(audio=0))

    with pytest.raises(AssemblyError, match="audio topology"):
        assembly._postflight_output(
            Path("film.mp4"), expected_duration=10.0, expect_audio=True
        )


def test_postflight_rejects_unexpected_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assembly, "probe_media", lambda _path: _media(audio=1))

    with pytest.raises(AssemblyError, match="audio topology"):
        assembly._postflight_output(
            Path("film.mp4"), expected_duration=10.0, expect_audio=False
        )


def test_postflight_rejects_multiple_video_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        assembly,
        "probe_media",
        lambda _path: _media(video=2, frame_counts=[240, 240]),
    )

    with pytest.raises(AssemblyError, match="exactly one video stream"):
        assembly._postflight_output(
            Path("film.mp4"), expected_duration=10.0, expect_audio=False
        )


def test_postflight_rejects_missing_decoded_frame_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = _media()
    media.pop("video_frame_counts")
    monkeypatch.setattr(assembly, "probe_media", lambda _path: media)

    with pytest.raises(AssemblyError, match="decoded-frame evidence"):
        assembly._postflight_output(
            Path("film.mp4"), expected_duration=10.0, expect_audio=False
        )


@pytest.mark.parametrize("frame_counts", [[], [None], [0], [-1], [240, 240]])
def test_postflight_rejects_invalid_decoded_frame_evidence(
    monkeypatch: pytest.MonkeyPatch,
    frame_counts: list[int | None],
) -> None:
    monkeypatch.setattr(
        assembly,
        "probe_media",
        lambda _path: _media(frame_counts=frame_counts),
    )

    with pytest.raises(AssemblyError, match="decoded-frame evidence"):
        assembly._postflight_output(
            Path("film.mp4"), expected_duration=10.0, expect_audio=False
        )
