"""Training resource and experiment budget controls for CINEOS evolution loops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingBudget:
    max_generations: int = 5
    max_candidates_per_generation: int = 4
    max_total_candidates: int = 20
    max_gpu_hours: float = 24.0
    early_stop_patience: int = 2
    minimum_improvement: float = 0.002

    def __post_init__(self) -> None:
        if self.max_generations < 1:
            raise ValueError("max_generations must be at least 1")
        if self.max_candidates_per_generation < 1:
            raise ValueError("max_candidates_per_generation must be at least 1")
        if self.max_total_candidates < 1:
            raise ValueError("max_total_candidates must be at least 1")
        if self.max_gpu_hours <= 0:
            raise ValueError("max_gpu_hours must be positive")
        if self.early_stop_patience < 1:
            raise ValueError("early_stop_patience must be at least 1")
        if self.minimum_improvement < 0:
            raise ValueError("minimum_improvement must be non-negative")


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    generations_completed: int = 0
    candidates_trained: int = 0
    gpu_hours_used: float = 0.0
    consecutive_non_improving_generations: int = 0


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    allowed: bool
    reason: str


class ExperimentBudgetController:
    """Decide whether an automated evolution run may continue safely."""

    def __init__(self, budget: TrainingBudget) -> None:
        self.budget = budget

    def may_start_generation(self, usage: ResourceUsage) -> BudgetDecision:
        if usage.generations_completed >= self.budget.max_generations:
            return BudgetDecision(False, "generation budget exhausted")
        if usage.candidates_trained >= self.budget.max_total_candidates:
            return BudgetDecision(False, "candidate budget exhausted")
        if usage.gpu_hours_used >= self.budget.max_gpu_hours:
            return BudgetDecision(False, "GPU-hour budget exhausted")
        if (
            usage.consecutive_non_improving_generations
            >= self.budget.early_stop_patience
        ):
            return BudgetDecision(False, "early stopping patience exhausted")
        return BudgetDecision(True, "budget available")

    def candidate_allowance(self, usage: ResourceUsage) -> int:
        remaining_total = self.budget.max_total_candidates - usage.candidates_trained
        if remaining_total <= 0:
            return 0
        return min(self.budget.max_candidates_per_generation, remaining_total)

    def update_after_generation(
        self,
        usage: ResourceUsage,
        *,
        candidates_trained: int,
        gpu_hours: float,
        improvement: float,
    ) -> ResourceUsage:
        if candidates_trained < 0 or gpu_hours < 0:
            raise ValueError("resource usage deltas must be non-negative")
        non_improving = (
            0
            if improvement >= self.budget.minimum_improvement
            else usage.consecutive_non_improving_generations + 1
        )
        return ResourceUsage(
            generations_completed=usage.generations_completed + 1,
            candidates_trained=usage.candidates_trained + candidates_trained,
            gpu_hours_used=usage.gpu_hours_used + gpu_hours,
            consecutive_non_improving_generations=non_improving,
        )
