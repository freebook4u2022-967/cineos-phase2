"""Multi-generation model evolution controller for CINEOS native image research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .checkpoint_gate import CheckpointBenchmarkGate, CheckpointScore
from .evolution_search import EvolutionConfig, mutate_config
from .model_tournament import AutomatedModelTournament, TournamentCandidate


EvolutionTrainer = Callable[[EvolutionConfig, Path], TournamentCandidate]


@dataclass(frozen=True, slots=True)
class EvolutionGeneration:
    generation: int
    candidates: tuple[EvolutionConfig, ...]
    winner_id: str
    promoted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class EvolutionRunResult:
    generations: tuple[EvolutionGeneration, ...]
    best_score: CheckpointScore


@dataclass(slots=True)
class MultiGenerationEvolutionController:
    gate: CheckpointBenchmarkGate
    candidates_per_generation: int = 3

    def __post_init__(self) -> None:
        if self.candidates_per_generation < 1:
            raise ValueError("candidates_per_generation must be at least 1")

    def run(
        self,
        seed_config: EvolutionConfig,
        trainer: EvolutionTrainer,
        work_dir: str | Path,
        *,
        generations: int = 3,
        incumbent: CheckpointScore | None = None,
    ) -> EvolutionRunResult:
        if generations < 1:
            raise ValueError("generations must be at least 1")
        destination = Path(work_dir)
        destination.mkdir(parents=True, exist_ok=True)
        current = seed_config
        best = incumbent
        history = []
        tournament = AutomatedModelTournament(self.gate)

        for generation in range(1, generations + 1):
            configs = [current]
            while len(configs) < self.candidates_per_generation:
                configs.append(mutate_config(current, generation + len(configs)))
            config_by_id = {config.candidate_id: config for config in configs}

            def train_candidate(candidate_id: str, candidate_dir: Path):
                return trainer(config_by_id[candidate_id], candidate_dir)

            result = tournament.run(
                tuple(config_by_id),
                train_candidate,
                destination / f"generation-{generation}",
                incumbent=best,
            )
            history.append(
                EvolutionGeneration(
                    generation=generation,
                    candidates=tuple(configs),
                    winner_id=result.winner.candidate_id,
                    promoted=result.promoted,
                    reason=result.reason,
                )
            )
            if result.promoted:
                best = result.winner.score
                current = config_by_id[result.winner.candidate_id]
            current = mutate_config(current, generation)

        if best is None:
            raise RuntimeError("evolution run completed without an eligible checkpoint")
        return EvolutionRunResult(tuple(history), best)
