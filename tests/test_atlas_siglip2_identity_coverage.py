from __future__ import annotations

import pytest

from cineos.atlas.siglip2_video_scorer import (
    SigLIP2VideoScorerError,
    _identity_score,
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


def test_multi_reference_identity_passes_when_every_identity_has_frame_support() -> None:
    score = _identity_score(
        [(1.0, 0.0), (0.0, 1.0)],
        [(1.0, 0.0), (0.0, 1.0)],
        mean_weight=0.7,
    )

    assert score == pytest.approx(1.0)


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
