"""Multi-generation model evolution controller for CINEOS native image research."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .checkpoint_gate import CheckpointBenchmarkGate, CheckpointScore
from .evolution_search import EvolutionConfig, mutate_config
from .evolution_state import (
    CandidateProgress,
    EvolutionResumeState,
    EvolutionStateStore,
    recover_interrupted_candidates,
)
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

    def _generation_configs(
        self,
        current: EvolutionConfig,
        generation: int,
    ) -> tuple[EvolutionConfig, ...]:
        configs = [current]
        while len(configs) < self.candidates_per_generation:
            configs.append(mutate_config(current, generation + len(configs)))
        return tuple(configs)

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
            configs = self._generation_configs(current, generation)
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
                    candidates=configs,
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

    def run_resumable(
        self,
        seed_config: EvolutionConfig,
        trainer: EvolutionTrainer,
        work_dir: str | Path,
        state_store: EvolutionStateStore,
        *,
        run_id: str,
        generations: int = 3,
        incumbent: CheckpointScore | None = None,
    ) -> EvolutionRunResult:
        """Run evolution with atomic candidate-level recovery checkpoints."""
        if generations < 1:
            raise ValueError("generations must be at least 1")
        destination = Path(work_dir)
        destination.mkdir(parents=True, exist_ok=True)

        if state_store.exists():
            state = recover_interrupted_candidates(state_store.load())
            if state.run_id != run_id:
                raise ValueError("resume state belongs to a different evolution run")
            state_store.save(state)
        else:
            state = EvolutionResumeState(
                run_id=run_id,
                current_generation=1,
                current_config=seed_config,
                best_score=incumbent,
            )
            state_store.save(state)

        current = state.current_config or seed_config
        best = state.best_score if state.best_score is not None else incumbent
        history = []

        for generation in range(state.current_generation, generations + 1):
            configs = self._generation_configs(current, generation)
            config_by_id = {config.candidate_id: config for config in configs}
            existing_ids = {item.candidate_id for item in state.candidates}
            if not state.candidates:
                state.candidates = tuple(
                    CandidateProgress(config.candidate_id) for config in configs
                )
                state_store.save(state)
            elif existing_ids != set(config_by_id):
                raise ValueError("resume candidate set does not match deterministic generation")

            completed = []
            generation_dir = destination / f"generation-{generation}"
            for config in configs:
                progress = state.candidate(config.candidate_id)
                if progress is None:
                    raise RuntimeError("candidate progress missing from resume state")
                if progress.status == "completed":
                    if progress.score is None or progress.checkpoint_path is None:
                        raise RuntimeError("completed candidate has incomplete resume data")
                    completed.append(
                        TournamentCandidate(
                            config.candidate_id,
                            progress.checkpoint_path,
                            progress.score,
                        )
                    )
                    continue

                candidate_dir = generation_dir / config.candidate_id
                state.with_candidate(
                    CandidateProgress(
                        candidate_id=config.candidate_id,
                        status="running",
                        checkpoint_path=progress.checkpoint_path,
                        error=progress.error,
                    )
                )
                state_store.save(state)

                candidate = trainer(config, candidate_dir)
                if candidate.candidate_id != config.candidate_id:
                    raise ValueError("trainer returned a mismatched candidate id")
                state.with_candidate(
                    CandidateProgress(
                        candidate_id=config.candidate_id,
                        status="completed",
                        checkpoint_path=candidate.checkpoint_path,
                        score=candidate.score,
                    )
                )
                state_store.save(state)
                completed.append(candidate)

            ranked = tuple(
                sorted(
                    completed,
                    key=lambda item: item.score.quality_score,
                    reverse=True,
                )
            )
            winner = ranked[0]
            decision = self.gate.promote_if_better(
                winner.score,
                best,
                generation_dir / "best_checkpoint.json",
            )
            history.append(
                EvolutionGeneration(
                    generation=generation,
                    candidates=configs,
                    winner_id=winner.candidate_id,
                    promoted=decision.promoted,
                    reason=decision.reason,
                )
            )
            if decision.promoted:
                best = winner.score
                current = config_by_id[winner.candidate_id]
            current = mutate_config(current, generation)

            state.best_score = best
            state.current_config = current
            state.current_generation = generation + 1
            state.completed_generations = tuple(
                sorted(set((*state.completed_generations, generation)))
            )
            state.candidates = ()
            state_store.save(state)

        if best is None:
            raise RuntimeError("evolution run completed without an eligible checkpoint")
        return EvolutionRunResult(tuple(history), best)
