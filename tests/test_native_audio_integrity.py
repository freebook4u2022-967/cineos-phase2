from __future__ import annotations

from pathlib import Path

import pytest

from cineos.native_video.audio_integrity import (
    AudioIntegrityPolicy,
    AudioStreamEvidence,
    FinalFilmAudioIntegrityGate,
)


class _Inspector:
    def __init__(self, stream: AudioStreamEvidence | None) -> None:
        self.stream = stream
        self.calls: list[Path] = []

    def inspect(self, movie_path: str | Path) -> AudioStreamEvidence | None:
        self.calls.append(Path(movie_path))
        return self.stream


def _stream(
    *, sample_rate_hz: int = 48000, channels: int = 2, duration_seconds: float = 10.0
) -> AudioStreamEvidence:
    return AudioStreamEvidence(
        codec_name="aac",
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        duration_seconds=duration_seconds,
    )


def test_required_audio_rejects_missing_stream(tmp_path) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"film")
    inspector = _Inspector(None)
    gate = FinalFilmAudioIntegrityGate(inspector=inspector)

    report = gate.evaluate(movie, expected_duration_seconds=10.0, required=True)

    assert report.decision == "reject"
    assert report.accepted is False
    assert report.stream is None
    assert report.directives == (
        "restore or render the required final-film audio stream",
    )
    assert inspector.calls == [movie]


def test_optional_silent_film_accepts_missing_stream(tmp_path) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"film")
    gate = FinalFilmAudioIntegrityGate(inspector=_Inspector(None))

    report = gate.evaluate(movie, expected_duration_seconds=8.0, required=False)

    assert report.decision == "accept"
    assert report.required is False
    assert report.duration_delta_seconds is None


def test_audio_gate_rejects_weak_stream_and_timeline_drift(tmp_path) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"film")
    gate = FinalFilmAudioIntegrityGate(
        policy=AudioIntegrityPolicy(
            min_sample_rate_hz=24000,
            min_channels=2,
            max_duration_delta_seconds=0.25,
        ),
        inspector=_Inspector(
            _stream(sample_rate_hz=16000, channels=1, duration_seconds=8.5)
        ),
    )

    report = gate.evaluate(movie, expected_duration_seconds=10.0)

    assert report.decision == "reject"
    assert report.duration_delta_seconds == pytest.approx(1.5)
    assert report.directives == (
        "raise final-film audio sample rate to at least 24000 Hz",
        "encode at least 2 final-film audio channel(s)",
        "realign final-film audio duration with the authored picture timeline",
    )


def test_audio_gate_accepts_healthy_aligned_stream(tmp_path) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"film")
    gate = FinalFilmAudioIntegrityGate(inspector=_Inspector(_stream(duration_seconds=9.8)))

    report = gate.evaluate(movie, expected_duration_seconds=10.0)

    assert report.decision == "accept"
    assert report.accepted is True
    assert report.duration_delta_seconds == pytest.approx(0.2)
    assert report.as_dict()["stream"]["codec_name"] == "aac"


@pytest.mark.parametrize("expected", [0.0, -1.0, float("nan"), float("inf")])
def test_audio_gate_rejects_invalid_expected_duration(tmp_path, expected: float) -> None:
    movie = tmp_path / "film.mp4"
    movie.write_bytes(b"film")
    gate = FinalFilmAudioIntegrityGate(inspector=_Inspector(_stream()))

    with pytest.raises(ValueError, match="finite and positive"):
        gate.evaluate(movie, expected_duration_seconds=expected)


def test_audio_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="min_sample_rate_hz"):
        AudioIntegrityPolicy(min_sample_rate_hz=0)
    with pytest.raises(ValueError, match="min_channels"):
        AudioIntegrityPolicy(min_channels=0)
    with pytest.raises(ValueError, match="max_duration_delta_seconds"):
        AudioIntegrityPolicy(max_duration_delta_seconds=float("nan"))
