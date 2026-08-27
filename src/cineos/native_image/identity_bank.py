"""Character identity embedding bank for CINEOS continuity training.

The identity bank is deliberately framework-neutral so the same acceptance and
continuity semantics are available during CPU validation and neural training.  In
addition to the legacy centroid builder, production callers can use robust
multi-reference fusion to reject inconsistent approved-reference embeddings before
an identity anchor is committed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CharacterIdentityEmbedding:
    character_id: str
    vector: tuple[float, ...]
    reference_count: int


class CharacterIdentityEmbeddingBank:
    """Aggregate reference embeddings into a stable normalized identity vector.

    The bank intentionally uses only Python numeric primitives so identity metadata,
    QC, and benchmarking remain available in lightweight CINEOS installations.
    Tensor conversion belongs at the neural-training boundary.
    """

    def __init__(self) -> None:
        self._entries: dict[str, CharacterIdentityEmbedding] = {}

    @staticmethod
    def _normalize(vector) -> tuple[float, ...]:
        values = tuple(float(value) for value in vector)
        if not values:
            raise ValueError("identity embedding must not be empty")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("identity embedding must contain only finite values")
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("identity embedding must not be zero")
        return tuple(value / norm for value in values)

    @staticmethod
    def _validate_dimensions(vectors: Sequence[tuple[float, ...]]) -> int:
        if not vectors:
            raise ValueError("at least one reference embedding is required")
        width = len(vectors[0])
        if any(len(vector) != width for vector in vectors):
            raise ValueError("reference embeddings must share the same dimension")
        return width

    @classmethod
    def _weighted_centroid(
        cls,
        vectors: Sequence[tuple[float, ...]],
        weights: Sequence[float],
    ) -> tuple[float, ...]:
        width = cls._validate_dimensions(vectors)
        if len(vectors) != len(weights):
            raise ValueError("reference weights must match reference embeddings")
        numeric_weights = tuple(float(weight) for weight in weights)
        if any(
            not math.isfinite(weight) or weight <= 0.0 for weight in numeric_weights
        ):
            raise ValueError("reference weights must be finite and positive")
        total = sum(numeric_weights)
        centroid = tuple(
            sum(
                vector[index] * weight
                for vector, weight in zip(vectors, numeric_weights, strict=True)
            )
            / total
            for index in range(width)
        )
        return cls._normalize(centroid)

    def build_character(
        self, character_id: str, reference_vectors
    ) -> CharacterIdentityEmbedding:
        """Build the backwards-compatible equal-weight normalized centroid."""
        if not character_id.strip():
            raise ValueError("character_id must not be empty")
        vectors = tuple(self._normalize(vector) for vector in reference_vectors)
        self._validate_dimensions(vectors)
        centroid = self._weighted_centroid(vectors, (1.0,) * len(vectors))
        entry = CharacterIdentityEmbedding(
            character_id=character_id,
            vector=centroid,
            reference_count=len(vectors),
        )
        self._entries[character_id] = entry
        return entry

    def build_character_robust(
        self,
        character_id: str,
        reference_vectors,
        *,
        reference_weights: Sequence[float] | None = None,
        min_consensus_similarity: float = 0.60,
        minimum_references: int = 2,
    ) -> CharacterIdentityEmbedding:
        """Fuse approved references while rejecting identity-inconsistent outliers.

        Each reference is normalized and scored by its mean cosine similarity to all
        other references.  A reference below ``min_consensus_similarity`` is excluded
        before the final weighted centroid is committed.  This protects the durable
        character anchor from a mislabeled, corrupted, or visually inconsistent
        approved reference without coupling CINEOS to any particular face encoder.

        The method fails closed when too few mutually-consistent references survive.
        For a deliberate one-reference identity, callers should use ``build_character``
        rather than silently disabling consensus validation.
        """
        if not character_id.strip():
            raise ValueError("character_id must not be empty")
        if not -1.0 <= min_consensus_similarity <= 1.0:
            raise ValueError("min_consensus_similarity must be between -1 and 1")
        if minimum_references < 2:
            raise ValueError("minimum_references must be at least 2 for robust fusion")

        vectors = tuple(self._normalize(vector) for vector in reference_vectors)
        self._validate_dimensions(vectors)
        if len(vectors) < minimum_references:
            raise ValueError("insufficient references for robust identity fusion")

        if reference_weights is None:
            weights = (1.0,) * len(vectors)
        else:
            weights = tuple(float(weight) for weight in reference_weights)
            if len(weights) != len(vectors):
                raise ValueError("reference weights must match reference embeddings")
            if any(not math.isfinite(weight) or weight <= 0.0 for weight in weights):
                raise ValueError("reference weights must be finite and positive")

        consensus_scores = []
        for index, vector in enumerate(vectors):
            similarities = [
                sum(left * right for left, right in zip(vector, other, strict=True))
                for other_index, other in enumerate(vectors)
                if other_index != index
            ]
            consensus_scores.append(sum(similarities) / len(similarities))

        accepted_indices = tuple(
            index
            for index, score in enumerate(consensus_scores)
            if score >= min_consensus_similarity
        )
        if len(accepted_indices) < minimum_references:
            raise ValueError("approved references do not meet identity consensus")

        accepted_vectors = tuple(vectors[index] for index in accepted_indices)
        accepted_weights = tuple(weights[index] for index in accepted_indices)
        centroid = self._weighted_centroid(accepted_vectors, accepted_weights)
        entry = CharacterIdentityEmbedding(
            character_id=character_id,
            vector=centroid,
            reference_count=len(accepted_vectors),
        )
        self._entries[character_id] = entry
        return entry

    def get(self, character_id: str) -> CharacterIdentityEmbedding:
        return self._entries[character_id]

    def characters(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def similarity(self, character_id: str, candidate_vector) -> float:
        anchor = self.get(character_id).vector
        candidate = self._normalize(candidate_vector)
        if len(anchor) != len(candidate):
            raise ValueError("candidate embedding dimension mismatch")
        return sum(left * right for left, right in zip(anchor, candidate, strict=True))
