from __future__ import annotations

from pathlib import Path

import pytest

from cineos.film import production_assembly
from cineos.film.exceptions import AssemblyError


def _media(*, frame_count: int | None = 48) -> dict[str, object]:
    return {
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "video_stream_count": 1,
        "video_codecs": ["h264"],
        "video_dimensions": [{"width": 1920, "height": 1080}],
        "video_frame_rates": ["24/1"],
        "video_frame_counts": [frame_count],
        "audio_stream_count": 0,
        "audio_codecs": [],
        "duration_seconds": 2.0,
    }


def _validate(monkeypatch: pytest.MonkeyPatch, media: dict[str, object]) -> dict[str, object]:
    monkeypatch.setattr(production_assembly, "probe_media", lambda _movie: media)
    return production_assembly._validate_final_media(
        Path("final.mp4"),
        audio_required=False,
        expected_duration=2.0,
        expected_width=1920,
        expected_height=1080,
        expected_frame_rate="24/1",
    )


def test_final_frame_count_matches_approved_timeline(monkeypatch: pytest.MonkeyPatch) -> None:
    validated = _validate(monkeypatch, _media(frame_count=48))

    timeline = validated["production_video_timeline"]
    assert timeline["decoded_frame_count"] == 48
    assert timeline["expected_decoded_frame_count"] == 48
    assert timeline["decoded_frame_count_tolerance"] == 1


def test_final_frame_count_allows_single_frame_boundary_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validated = _validate(monkeypatch, _media(frame_count=47))

    assert validated["production_video_timeline"]["decoded_frame_count"] == 47


@pytest.mark.parametrize("frame_count", [46, 50])
def test_final_frame_count_rejects_drift_beyond_boundary_tolerance(
    monkeypatch: pytest.MonkeyPatch,
    frame_count: int,
) -> None:
    with pytest.raises(AssemblyError, match="decoded frame count does not match"):
        _validate(monkeypatch, _media(frame_count=frame_count))


@pytest.mark.parametrize("frame_count", [None, 0])
def test_final_frame_count_rejects_unverified_count(
    monkeypatch: pytest.MonkeyPatch,
    frame_count: int | None,
) -> None:
    with pytest.raises(AssemblyError, match="invalid decoded frame-count evidence"):
        _validate(monkeypatch, _media(frame_count=frame_count))


def test_source_shot_does_not_assume_cfr_from_decoded_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media = _media(frame_count=None)
    monkeypatch.setattr(production_assembly, "probe_media", lambda _movie: media)

    validated = production_assembly._validate_bound_shot_media(
        "shot-vfr", Path("source.mp4")
    )

    assert validated["frame_rate"] == "24/1"
    assert validated["duration_seconds"] == 2.0
