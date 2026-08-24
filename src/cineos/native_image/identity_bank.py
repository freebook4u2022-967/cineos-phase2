"""Character identity embedding bank for CINEOS continuity training."""

from __future__ import annotations

import math
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
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise ValueError("identity embedding must not be zero")
        return tuple(value / norm for value in values)

    def build_character(self, character_id: str, reference_vectors) -> CharacterIdentityEmbedding:
        if not character_id.strip():
            raise ValueError("character_id must not be empty")
        vectors = tuple(self._normalize(vector) for vector in reference_vectors)
        if not vectors:
            raise ValueError("at least one reference embedding is required")
        width = len(vectors[0])
        if any(len(vector) != width for vector in vectors):
            raise ValueError("reference embeddings must share the same dimension")
        centroid = tuple(
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(width)
        )
        centroid = self._normalize(centroid)
        entry = CharacterIdentityEmbedding(
            character_id=character_id,
            vector=centroid,
            reference_count=len(vectors),
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
