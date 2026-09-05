"""Conditional response evaluation for CINEOS native image generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .latent_generation import ConditionalLatentGenerator
from .neural_backend import _load_torch


@dataclass(frozen=True, slots=True)
class ConditionalResponseReport:
    same_character_scene_distance: float
    different_character_same_scene_distance: float
    identity_consistency_score: float


def _cosine_similarity(first, second) -> float:
    torch = _load_torch()
    a = first.detach().float().reshape(-1).cpu()
    b = second.detach().float().reshape(-1).cpu()
    denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    if denominator == 0.0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(torch.dot(a, b) / denominator)


def _distance(first, second) -> float:
    return max(0.0, 1.0 - _cosine_similarity(first, second))


def identity_consistency_score(latents: tuple[object, ...]) -> float:
    if len(latents) < 2:
        raise ValueError("identity consistency requires at least two generated latents")
    similarities = []
    for index in range(len(latents) - 1):
        similarities.append(_cosine_similarity(latents[index], latents[index + 1]))
    return max(0.0, min(1.0, sum(similarities) / len(similarities)))


@dataclass(slots=True)
class ConditionalResponseEvaluator:
    generator: ConditionalLatentGenerator

    def evaluate(
        self,
        character_a: tuple[str | Path, ...],
        character_b: tuple[str | Path, ...],
        *,
        scene_a: tuple[str, str],
        scene_b: tuple[str, str],
        seed: int = 0,
    ) -> ConditionalResponseReport:
        a_scene_a = self.generator.sample_latent(
            character_a, scene_a[0], scene_a[1], seed=seed
        )
        a_scene_b = self.generator.sample_latent(
            character_a, scene_b[0], scene_b[1], seed=seed
        )
        b_scene_a = self.generator.sample_latent(
            character_b, scene_a[0], scene_a[1], seed=seed
        )
        repeated = tuple(
            self.generator.sample_latent(
                character_a,
                scene_a[0],
                scene_a[1],
                seed=seed + offset,
            )
            for offset in range(3)
        )
        values = (
            _distance(a_scene_a, a_scene_b),
            _distance(a_scene_a, b_scene_a),
            identity_consistency_score(repeated),
        )
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError("conditional evaluation produced non-finite metrics")
        return ConditionalResponseReport(*values)
