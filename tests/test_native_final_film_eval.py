from __future__ import annotations

import pytest

from cineos.native_video.final_eval import (
    TemporalFilmEvalPolicy,
    evaluate_sampled_frames,
)


def _frame(value: int, size: int = 16) -> bytes:
    return bytes([value]) * size


def test_final_film_eval_accepts_non_black_motion_evidence() -> None:
    report = evaluate_sampled_frames(
        (_frame(40), _frame(50), _frame(61), _frame(73))
    )

    assert report.decision == "accept"
    assert report.accepted is True
    assert report.black_frame_ratio == 0.0
    assert report.frozen_transition_ratio == 0.0
    assert 10.0 < report.mean_interframe_mad < 12.0


def test_final_film_eval_rejects_black_regions() -> None:
    report = evaluate_sampled_frames(
        (_frame(0), _frame(0), _frame(0), _frame(70)),
        TemporalFilmEvalPolicy(max_black_ratio=0.25, max_frozen_ratio=1.0),
    )

    assert report.decision == "reject"
    assert report.accepted is False
    assert report.black_frame_ratio == pytest.approx(0.75)
    assert any("black" in directive for directive in report.directives)


def test_final_film_eval_rejects_frozen_sequence() -> None:
    report = evaluate_sampled_frames(
        (_frame(100), _frame(100), _frame(100), _frame(100))
    )

    assert report.decision == "reject"
    assert report.frozen_transition_ratio == 1.0
    assert any("frozen" in directive for directive in report.directives)


def test_final_film_eval_warns_on_excessive_hard_cuts() -> None:
    report = evaluate_sampled_frames(
        (_frame(20), _frame(100), _frame(20), _frame(100)),
        TemporalFilmEvalPolicy(max_frozen_ratio=1.0),
    )

    assert report.decision == "warn"
    assert report.accepted is True
    assert report.hard_cut_transition_ratio == 1.0
    assert any("hard-cut" in directive for directive in report.directives)


def test_final_film_eval_requires_consistent_decoded_frame_size() -> None:
    with pytest.raises(ValueError, match="same non-zero size"):
        evaluate_sampled_frames((b"abc", b"abcd"))
