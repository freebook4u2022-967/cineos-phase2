"""Learned semantic scoring bridge for production video QC.

This module supplies the missing semantic half of :mod:`artifact_video_observer`.
It does not pretend pixel heuristics can prove character identity or motion quality.
Instead it requires real embedding and motion backends, combines their measured
outputs conservatively, and exposes only normalized metrics that the artifact-bound
quality gate can audit and feed into reject/rerender decisions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .artifact_video_observer import RGBVideoSample


class SemanticVideoScorerError(RuntimeError):
    """Raised when learned semantic evidence is missing or malformed."""


Embedding = Sequence[float]
FrameEmbedder = Callable[[RGBVideoSample], Sequence[Embedding]]
ReferenceEmbeddingLoader = Callable[[str], Embedding]
MotionScorer = Callable[..., float]


def _validated_embedding(value: Embedding, *, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise SemanticVideoScorerError(f"{label} must be a numeric embedding")
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise SemanticVideoScorerError(f"{label} must be a numeric embedding") from exc
    if not vector:
        raise SemanticVideoScorerError(f"{label} must not be empty")
    if any(not math.isfinite(item) for item in vector):
        raise SemanticVideoScorerError(f"{label} contains non-finite values")
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 0.0:
        raise SemanticVideoScorerError(f"{label} must have non-zero magnitude")
    return tuple(item / norm for item in vector)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise SemanticVideoScorerError(
            f"embedding dimension mismatch: {len(left)} != {len(right)}"
        )
    return sum(a * b for a, b in zip(left, right, strict=True))


def _normalized_similarity(cosine: float) -> float:
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


class LearnedIdentityMotionScorer:
    """Measure identity persistence plus learned motion quality for one shot.

    ``frame_embedder`` must return one learned visual embedding per decoded frame.
    ``reference_embedding_loader`` resolves every approved CINEOS reference id to
    an embedding produced in the same feature space. ``motion_scorer`` is kept
    separate on purpose: identity embeddings are not relabeled as motion evidence.

    Identity uses a conservative blend of mean and worst-frame similarity. This
    makes a brief face/character drift visible instead of letting many good frames
    fully hide it. Multiple approved references are supported by taking the best
    matching approved reference for each generated frame.
    """

    semantic_measurement_evidence = True

    def __init__(
        self,
        frame_embedder: FrameEmbedder,
        reference_embedding_loader: ReferenceEmbeddingLoader,
        motion_scorer: MotionScorer,
        *,
        mean_weight: float = 0.7,
    ) -> None:
        if not callable(frame_embedder):
            raise TypeError("frame_embedder must be callable")
        if not callable(reference_embedding_loader):
            raise TypeError("reference_embedding_loader must be callable")
        if not callable(motion_scorer):
            raise TypeError("motion_scorer must be callable")
        if not 0.0 <= mean_weight <= 1.0:
            raise ValueError("mean_weight must be between 0 and 1")
        self.frame_embedder = frame_embedder
        self.reference_embedding_loader = reference_embedding_loader
        self.motion_scorer = motion_scorer
        self.mean_weight = float(mean_weight)

    def __call__(
        self,
        sample: RGBVideoSample,
        *,
        artifact: Path,
        shot: Any,
        attempt_index: int,
    ) -> dict[str, float]:
        reference_ids = getattr(shot, "approved_reference_ids", None)
        if not isinstance(reference_ids, list) or not reference_ids:
            raise SemanticVideoScorerError(
                "semantic identity scoring requires approved_reference_ids"
            )

        references = [
            _validated_embedding(
                self.reference_embedding_loader(reference_id),
                label=f"reference embedding {reference_id!r}",
            )
            for reference_id in reference_ids
        ]

        raw_frames = self.frame_embedder(sample)
        if isinstance(raw_frames, (str, bytes)):
            raise SemanticVideoScorerError("frame_embedder must return embeddings")
        frames = [
            _validated_embedding(value, label=f"frame embedding {index}")
            for index, value in enumerate(raw_frames)
        ]
        if len(frames) != len(sample.frames):
            raise SemanticVideoScorerError(
                "frame_embedder must return exactly one embedding per sampled frame"
            )

        frame_scores: list[float] = []
        for frame in frames:
            best = max(
                _normalized_similarity(_cosine(frame, ref)) for ref in references
            )
            frame_scores.append(best)

        mean_score = sum(frame_scores) / len(frame_scores)
        worst_score = min(frame_scores)
        identity_similarity = (
            self.mean_weight * mean_score + (1.0 - self.mean_weight) * worst_score
        )

        motion = self.motion_scorer(
            sample,
            artifact=artifact,
            shot=shot,
            attempt_index=attempt_index,
        )
        if isinstance(motion, bool) or not isinstance(motion, (int, float)):
            raise SemanticVideoScorerError("motion_scorer must return a numeric score")
        motion_quality = float(motion)
        if not math.isfinite(motion_quality) or not 0.0 <= motion_quality <= 1.0:
            raise SemanticVideoScorerError(
                "motion_scorer score must be finite and between 0 and 1"
            )

        return {
            "identity_similarity": identity_similarity,
            "motion_quality": motion_quality,
        }


__all__ = [
    "Embedding",
    "FrameEmbedder",
    "LearnedIdentityMotionScorer",
    "MotionScorer",
    "ReferenceEmbeddingLoader",
    "SemanticVideoScorerError",
]
