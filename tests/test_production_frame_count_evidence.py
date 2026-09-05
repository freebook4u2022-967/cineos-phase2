from pathlib import Path

import pytest

from cineos.film.exceptions import AssemblyError
from cineos.film.production_assembly import _expected_timeline_frame_count


def _bound() -> list[tuple[str, Path, str, str]]:
    return [("shot-1", Path("shot-1.mp4"), "a" * 64, "b" * 64)]


def test_explicit_edit_rejects_known_frame_starvation() -> None:
    shot_media = {"shot-1": {"decoded_frame_count": 46}}

    with pytest.raises(
        AssemblyError,
        match="decoded frame count does not support approved edit duration",
    ):
        _expected_timeline_frame_count(
            _bound(),
            shot_media,
            frame_rate="24/1",
            durations=[2.0],
        )


def test_explicit_edit_allows_one_frame_probe_tolerance_without_lowering_expectation() -> None:
    shot_media = {"shot-1": {"decoded_frame_count": 47}}

    result = _expected_timeline_frame_count(
        _bound(),
        shot_media,
        frame_rate="24/1",
        durations=[2.0],
    )

    assert result["mode"] == "per-shot-cfr-hard-trim"
    assert result["expected_decoded_frame_count"] == 48
    assert result["expected_per_shot_decoded_frame_counts"] == [48]


def test_explicit_edit_keeps_approved_expectation_when_source_count_is_unavailable() -> None:
    shot_media = {"shot-1": {"decoded_frame_count": None}}

    result = _expected_timeline_frame_count(
        _bound(),
        shot_media,
        frame_rate="24/1",
        durations=[2.0],
    )

    assert result["expected_decoded_frame_count"] == 48
    assert result["expected_per_shot_decoded_frame_counts"] == [48]
