"""Budget-aware wrapper for resumable CINEOS evolution runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .checkpoint_gate import CheckpointScore
from .evolution_controller import EvolutionRunResult, MultiGenerationEvolutionController
from .evolution_search import EvolutionConfig
from .evolution_state import EvolutionStateStore
from .model_tournament import TournamentCandidate
from .training_budget import ExperimentBudgetController, ResourceUsage

MeasuredTrainer = Callable[[EvolutionConfig, Path], tuple[TournamentCandidate, float]]


@dataclass(frozen=True, slots=True)
class BudgetedEvolutionResult:
    evolution: EvolutionRunResult | None
    usage: ResourceUsage
    stopped_reason: str | None


@dataclass(slots=True)
class BudgetAwareEvolutionRuntime:
    controller: MultiGenerationEvolutionController
    budget: ExperimentBudgetController

    def run(
        self,
        seed_config: EvolutionConfig,
        trainer: MeasuredTrainer,
        work_dir: str | Path,
        state_store: EvolutionStateStore,
        *,
        run_id: str,
        incumbent: CheckpointScore | None = None,
    ) -> BudgetedEvolutionResult:
        state = state_store.load() if state_store.exists() else None
        usage = state.usage if state is not None else ResourceUsage()
        decision = self.budget.may_start_generation(usage)
        if not decision.allowed:
            return BudgetedEvolutionResult(None, usage, decision.reason)

        allowance = self.budget.candidate_allowance(usage)
        if allowance <= 0:
            return BudgetedEvolutionResult(None, usage, "candidate budget exhausted")
        original_candidates = self.controller.candidates_per_generation
        self.controller.candidates_per_generation = min(original_candidates, allowance)
        gpu_hours = 0.0
        trained = 0

        def measured(config: EvolutionConfig, directory: Path) -> TournamentCandidate:
            nonlocal gpu_hours, trained
            candidate, hours = trainer(config, directory)
            if hours < 0:
                raise ValueError("trainer GPU hours must be non-negative")
            projected = usage.gpu_hours_used + gpu_hours + hours
            if projected > self.budget.budget.max_gpu_hours:
                raise RuntimeError("candidate would exceed GPU-hour budget")
            gpu_hours += hours
            trained += 1
            return candidate

        try:
            before = incumbent.quality_score if incumbent is not None else 0.0
            result = self.controller.run_resumable(
                seed_config,
                measured,
                work_dir,
                state_store,
                run_id=run_id,
                generations=usage.generations_completed + 1,
                incumbent=incumbent,
            )
            improvement = max(0.0, result.best_score.quality_score - before)
            updated = self.budget.update_after_generation(
                usage,
                candidates_trained=trained,
                gpu_hours=gpu_hours,
                improvement=improvement,
            )
            state = state_store.load()
            state.usage = updated
            state_store.save(state)
            next_decision = self.budget.may_start_generation(updated)
            return BudgetedEvolutionResult(
                result,
                updated,
                None if next_decision.allowed else next_decision.reason,
            )
        finally:
            self.controller.candidates_per_generation = original_candidates
