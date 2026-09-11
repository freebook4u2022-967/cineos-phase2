from __future__ import annotations

import math

import pytest

from cineos.atlas.siglip2_video_scorer import (
    SigLIP2VideoScorerError,
    _identity_score,
    _multi_character_copresence_score,
)


def test_single_reference_identity_scoring_preserves_historical_aggregation() -> None:
    score = _identity_score(
        [(1.0, 0.0), (0.8, 0.6)],
        [(1.0, 0.0)],
        mean_weight=0.7,
    )

    assert score == pytest.approx(0.935)


def test_multi_reference_identity_is_capped_by_missing_identity_support() -> None:
    score = _identity_score(
        [(1.0, 0.0), (1.0, 0.0)],
        [(1.0, 0.0), (-1.0, 0.0)],
        mean_weight=0.7,
    )

    assert score == pytest.approx(0.0)


def test_multi_reference_identity_passes_when_every_identity_has_frame_support() -> (
    None
):
    score = _identity_score(
        [(1.0, 0.0), (0.0, 1.0)],
        [(1.0, 0.0), (0.0, 1.0)],
        mean_weight=0.7,
    )

    assert score == pytest.approx(1.0)


def test_multi_reference_identity_rejects_single_lucky_frame_support() -> None:
    score = _identity_score(
        [(1.0, 0.0)] * 7 + [(-1.0, 0.0)],
        [(1.0, 0.0), (-1.0, 0.0)],
        mean_weight=0.7,
        multi_identity_support_fraction=0.25,
    )

    assert score == pytest.approx(0.5)


def test_multi_reference_identity_accepts_sustained_fractional_support() -> None:
    score = _identity_score(
        [(1.0, 0.0)] * 6 + [(-1.0, 0.0)] * 2,
        [(1.0, 0.0), (-1.0, 0.0)],
        mean_weight=0.7,
        multi_identity_support_fraction=0.25,
    )

    assert score == pytest.approx(1.0)


def test_multi_reference_identity_rejects_invalid_support_fraction() -> None:
    with pytest.raises(SigLIP2VideoScorerError, match="support fraction"):
        _identity_score(
            [(1.0, 0.0), (-1.0, 0.0)],
            [(1.0, 0.0), (-1.0, 0.0)],
            mean_weight=0.7,
            multi_identity_support_fraction=0.0,
        )


def test_multi_character_copresence_rejects_disjoint_character_appearances() -> None:
    score = _multi_character_copresence_score(
        [(1.0, 0.0)] * 4 + [(0.0, 1.0)] * 4,
        [[(1.0, 0.0)], [(0.0, 1.0)]],
        support_fraction=0.25,
    )

    # Each character has excellent temporal support somewhere, but never on the
    # same frame. Legacy per-reference coverage alone would accept this pattern.
    assert score == pytest.approx(0.5)


def test_multi_character_copresence_accepts_sustained_shared_frame_support() -> None:
    shared = 1.0 / math.sqrt(2.0)
    score = _multi_character_copresence_score(
        [(shared, shared)] * 4 + [(1.0, 0.0)] * 4,
        [[(1.0, 0.0)], [(0.0, 1.0)]],
        support_fraction=0.5,
    )

    assert score == pytest.approx((shared + 1.0) / 2.0)


def test_multi_character_copresence_fails_closed_without_every_group() -> None:
    with pytest.raises(SigLIP2VideoScorerError, match="at least two character groups"):
        _multi_character_copresence_score(
            [(1.0, 0.0)],
            [[(1.0, 0.0)]],
            support_fraction=0.25,
        )


@pytest.mark.parametrize(
    ("frames", "references", "message"),
    [
        ([], [(1.0, 0.0)], "decoded frame features"),
        ([(1.0, 0.0)], [], "approved references"),
    ],
)
def test_identity_aggregation_fails_closed_without_required_evidence(
    frames: list[tuple[float, float]],
    references: list[tuple[float, float]],
    message: str,
) -> None:
    with pytest.raises(SigLIP2VideoScorerError, match=message):
        _identity_score(frames, references, mean_weight=0.7)
