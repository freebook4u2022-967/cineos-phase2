from pathlib import Path

from cineos.film.production_assembly import _expected_timeline_frame_count


def _bound(count: int = 5):
    return [
        (
            f"shot-{index}",
            Path(f"shot-{index}.mp4"),
            f"output-{index}",
            f"evidence-{index}",
        )
        for index in range(count)
    ]


def _media(counts):
    return {
        f"shot-{index}": {"decoded_frame_count": count}
        for index, count in enumerate(counts)
    }


def test_fractional_hard_trims_are_quantized_per_shot_not_after_duration_sum():
    result = _expected_timeline_frame_count(
        _bound(),
        _media([24] * 5),
        frame_rate="24/1",
        durations=[0.06] * 5,
    )

    # Each independent trim retains timestamps at 0 and 1/24s: 2 frames per shot.
    # Rounding the aggregate 0.30s timeline would incorrectly predict only 7 frames.
    assert result == {
        "mode": "per-shot-cfr-hard-trim",
        "expected_decoded_frame_count": 10,
        "expected_per_shot_decoded_frame_counts": [2, 2, 2, 2, 2],
    }


def test_exact_frame_boundary_does_not_gain_an_extra_frame():
    result = _expected_timeline_frame_count(
        _bound(),
        _media([24] * 5),
        frame_rate="48/2",
        durations=[0.125] * 5,
    )

    assert result["expected_decoded_frame_count"] == 15
    assert result["expected_per_shot_decoded_frame_counts"] == [3] * 5


def test_trim_expectation_is_capped_by_observed_source_frames():
    result = _expected_timeline_frame_count(
        _bound(),
        _media([1, 2, 3, 4, 5]),
        frame_rate="24/1",
        durations=[1.0] * 5,
    )

    assert result["expected_decoded_frame_count"] == 15
    assert result["expected_per_shot_decoded_frame_counts"] == [1, 2, 3, 4, 5]


def test_untrimmed_timeline_uses_observed_decoded_source_counts_directly():
    result = _expected_timeline_frame_count(
        _bound(),
        _media([47, 48, 49, 50, 51]),
        frame_rate="24/1",
        durations=None,
    )

    assert result == {
        "mode": "observed-source-decoded-frames",
        "expected_decoded_frame_count": 245,
        "expected_per_shot_decoded_frame_counts": [47, 48, 49, 50, 51],
    }


def test_untrimmed_timeline_keeps_missing_vfr_counts_explicitly_unverified():
    media = _media([24] * 5)
    media["shot-3"]["decoded_frame_count"] = None

    result = _expected_timeline_frame_count(
        _bound(),
        media,
        frame_rate="24/1",
        durations=None,
    )

    assert result == {
        "mode": "unverified-source-decoded-frames",
        "expected_decoded_frame_count": None,
        "expected_per_shot_decoded_frame_counts": None,
    }


def test_missing_frame_rate_does_not_invent_frame_count_evidence():
    result = _expected_timeline_frame_count(
        _bound(),
        _media([24] * 5),
        frame_rate=None,
        durations=[0.06] * 5,
    )

    assert result == {
        "mode": "unavailable-no-frame-rate",
        "expected_decoded_frame_count": None,
        "expected_per_shot_decoded_frame_counts": None,
    }
