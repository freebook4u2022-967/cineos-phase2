from cineos.native_image.training_budget import (
    ExperimentBudgetController,
    ResourceUsage,
    TrainingBudget,
)


def test_budget_blocks_after_generation_limit():
    controller = ExperimentBudgetController(TrainingBudget(max_generations=2))
    usage = ResourceUsage(generations_completed=2)
    decision = controller.may_start_generation(usage)
    assert decision.allowed is False
    assert "generation" in decision.reason


def test_budget_caps_candidates_per_generation_and_total():
    controller = ExperimentBudgetController(
        TrainingBudget(max_candidates_per_generation=4, max_total_candidates=5)
    )
    usage = ResourceUsage(candidates_trained=3)
    assert controller.candidate_allowance(usage) == 2


def test_gpu_budget_stops_training():
    controller = ExperimentBudgetController(TrainingBudget(max_gpu_hours=2.0))
    decision = controller.may_start_generation(ResourceUsage(gpu_hours_used=2.0))
    assert decision.allowed is False
    assert "GPU-hour" in decision.reason


def test_early_stopping_resets_after_meaningful_improvement():
    controller = ExperimentBudgetController(
        TrainingBudget(early_stop_patience=2, minimum_improvement=0.01)
    )
    usage = ResourceUsage(consecutive_non_improving_generations=1)
    improved = controller.update_after_generation(
        usage,
        candidates_trained=2,
        gpu_hours=0.5,
        improvement=0.02,
    )
    assert improved.consecutive_non_improving_generations == 0
    assert controller.may_start_generation(improved).allowed is True


def test_early_stopping_blocks_repeated_non_improvement():
    controller = ExperimentBudgetController(
        TrainingBudget(early_stop_patience=2, minimum_improvement=0.01)
    )
    usage = ResourceUsage()
    for _ in range(2):
        usage = controller.update_after_generation(
            usage,
            candidates_trained=1,
            gpu_hours=0.1,
            improvement=0.0,
        )
    assert controller.may_start_generation(usage).allowed is False
