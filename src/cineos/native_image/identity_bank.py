"""Character identity embedding bank for CINEOS continuity training."""

from __future__ import annotations

from dataclasses import dataclass

from .neural_backend import _load_torch


@dataclass(frozen=True, slots=True)
class CharacterIdentityEmbedding:
    character_id: str
    vector: tuple[float, ...]
    reference_count: int


class CharacterIdentityEmbeddingBank:
    """Aggregate multiple reference embeddings into a stable normalized identity vector."""

    def __init__(self) -> None:
        self._entries: dict[str, CharacterIdentityEmbedding] = {}

    @staticmethod
    def _normalize(vector):
        torch = _load_torch()
        tensor = torch.as_tensor(vector, dtype=torch.float32).reshape(-1)
        if tensor.numel() == 0:
            raise ValueError("identity embedding must not be empty")
        norm = torch.linalg.vector_norm(tensor)
        if float(norm) == 0.0:
            raise ValueError("identity embedding must not be zero")
        return tensor / norm

    def build_character(self, character_id: str, reference_vectors) -> CharacterIdentityEmbedding:
        torch = _load_torch()
        if not character_id.strip():
            raise ValueError("character_id must not be empty")
        vectors = tuple(self._normalize(vector) for vector in reference_vectors)
        if not vectors:
            raise ValueError("at least one reference embedding is required")
        width = vectors[0].numel()
        if any(vector.numel() != width for vector in vectors):
            raise ValueError("reference embeddings must share the same dimension")
        centroid = torch.stack(vectors).mean(dim=0)
        centroid = self._normalize(centroid)
        entry = CharacterIdentityEmbedding(
            character_id=character_id,
            vector=tuple(float(value) for value in centroid.tolist()),
            reference_count=len(vectors),
        )
        self._entries[character_id] = entry
        return entry

    def get(self, character_id: str) -> CharacterIdentityEmbedding:
        return self._entries[character_id]

    def characters(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def similarity(self, character_id: str, candidate_vector) -> float:
        torch = _load_torch()
        anchor = torch.tensor(self.get(character_id).vector, dtype=torch.float32)
        candidate = self._normalize(candidate_vector)
        if anchor.numel() != candidate.numel():
            raise ValueError("candidate embedding dimension mismatch")
        return float(torch.dot(anchor, candidate))
