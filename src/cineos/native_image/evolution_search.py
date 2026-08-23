"""Deterministic hyperparameter evolution search for CINEOS model tournaments."""

from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    learning_rate: float
    latent_dim: int
    hidden_dim: int
    flow_steps: int
    reconstruction_weight: float
    flow_weight: float

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.latent_dim <= 0 or self.hidden_dim <= 0 or self.flow_steps <= 0:
            raise ValueError("model dimensions and flow_steps must be positive")
        if self.reconstruction_weight < 0 or self.flow_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.reconstruction_weight + self.flow_weight == 0:
            raise ValueError("at least one loss weight must be positive")

    @property
    def candidate_id(self) -> str:
        return (
            f"lr{self.learning_rate:g}-z{self.latent_dim}-h{self.hidden_dim}-"
            f"s{self.flow_steps}-rw{self.reconstruction_weight:g}-"
            f"fw{self.flow_weight:g}"
        )


@dataclass(frozen=True, slots=True)
class EvolutionSearchSpace:
    learning_rates: tuple[float, ...]
    latent_dims: tuple[int, ...]
    hidden_dims: tuple[int, ...]
    flow_steps: tuple[int, ...]
    reconstruction_weights: tuple[float, ...]
    flow_weights: tuple[float, ...]

    def candidates(self, *, max_candidates: int | None = None) -> tuple[EvolutionConfig, ...]:
        configs = tuple(
            EvolutionConfig(*values)
            for values in itertools.product(
                self.learning_rates,
                self.latent_dims,
                self.hidden_dims,
                self.flow_steps,
                self.reconstruction_weights,
                self.flow_weights,
            )
        )
        if max_candidates is None:
            return configs
        if max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        return configs[:max_candidates]


def mutate_config(config: EvolutionConfig, generation: int) -> EvolutionConfig:
    """Create a bounded deterministic child configuration around a winner."""
    if generation < 1:
        raise ValueError("generation must be at least 1")
    direction = -1.0 if generation % 2 else 1.0
    lr_factor = 1.0 + (direction * min(0.25, 0.05 * generation))
    hidden_delta = 8 if generation % 2 else 16
    return EvolutionConfig(
        learning_rate=max(1e-6, config.learning_rate * lr_factor),
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim + hidden_delta,
        flow_steps=max(1, config.flow_steps + (1 if generation % 3 == 0 else 0)),
        reconstruction_weight=config.reconstruction_weight,
        flow_weight=config.flow_weight,
    )
