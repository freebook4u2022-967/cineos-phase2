from pathlib import Path

import pytest

from cineos.atlas.artifact_video_observer import RGBVideoSample
from cineos.atlas.semantic_video_scorer import (
    LearnedIdentityMotionScorer,
    SemanticVideoScorerError,
)


class Shot:
    approved_reference_ids = ["lead-front", "lead-profile"]


def _sample(frame_count: int = 3) -> RGBVideoSample:
    frame = bytes([32, 64, 96] * 4)
    return RGBVideoSample(width=2, height=2, frames=tuple(frame for _ in range(frame_count)))


def test_learned_semantic_scorer_uses_best_approved_reference_and_worst_frame():
    reference_embeddings = {
        "lead-front": (1.0, 0.0),
        "lead-profile": (0.0, 1.0),
    }
    frame_embeddings = (
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
    )
    observed = []

    def motion_scorer(sample, *, artifact, shot, attempt_index):
        observed.append((sample, artifact, shot, attempt_index))
        return 0.84

    scorer = LearnedIdentityMotionScorer(
        lambda _sample: frame_embeddings,
        reference_embeddings.__getitem__,
        motion_scorer,
        mean_weight=0.7,
    )

    metrics = scorer(
        _sample(),
        artifact=Path("candidate.mp4"),
        shot=Shot(),
        attempt_index=2,
    )

    # Per-frame best approved-reference scores are 1.0, 1.0 and 0.5.
    # The conservative blend is 70% mean + 30% worst frame.
    assert metrics["identity_similarity"] == pytest.approx(0.7 * (2.5 / 3.0) + 0.3 * 0.5)
    assert metrics["motion_quality"] == pytest.approx(0.84)
    assert observed[0][3] == 2


def test_learned_semantic_scorer_requires_one_embedding_per_sampled_frame():
    scorer = LearnedIdentityMotionScorer(
        lambda _sample: [(1.0, 0.0)],
        lambda _reference_id: (1.0, 0.0),
        lambda *_args, **_kwargs: 0.9,
    )

    with pytest.raises(SemanticVideoScorerError, match="exactly one embedding"):
        scorer(
            _sample(frame_count=2),
            artifact=Path("candidate.mp4"),
            shot=Shot(),
            attempt_index=0,
        )


def test_learned_semantic_scorer_rejects_embedding_dimension_mismatch():
    scorer = LearnedIdentityMotionScorer(
        lambda _sample: [(1.0, 0.0, 0.0)] * 3,
        lambda _reference_id: (1.0, 0.0),
        lambda *_args, **_kwargs: 0.9,
    )

    with pytest.raises(SemanticVideoScorerError, match="dimension mismatch"):
        scorer(
            _sample(),
            artifact=Path("candidate.mp4"),
            shot=Shot(),
            attempt_index=0,
        )


def test_learned_semantic_scorer_rejects_missing_approved_identity_references():
    class NoReferences:
        approved_reference_ids = []

    scorer = LearnedIdentityMotionScorer(
        lambda _sample: [(1.0, 0.0)] * 3,
        lambda _reference_id: (1.0, 0.0),
        lambda *_args, **_kwargs: 0.9,
    )

    with pytest.raises(SemanticVideoScorerError, match="approved_reference_ids"):
        scorer(
            _sample(),
            artifact=Path("candidate.mp4"),
            shot=NoReferences(),
            attempt_index=0,
        )


def test_learned_semantic_scorer_fails_closed_on_invalid_motion_score():
    scorer = LearnedIdentityMotionScorer(
        lambda _sample: [(1.0, 0.0)] * 3,
        lambda _reference_id: (1.0, 0.0),
        lambda *_args, **_kwargs: 1.2,
    )

    with pytest.raises(SemanticVideoScorerError, match="between 0 and 1"):
        scorer(
            _sample(),
            artifact=Path("candidate.mp4"),
            shot=Shot(),
            attempt_index=0,
        )
