from cineos.native_image.budget_runtime import BudgetAwareEvolutionRuntime
from cineos.native_image.checkpoint_gate import CheckpointBenchmarkGate, CheckpointScore
from cineos.native_image.evolution_controller import MultiGenerationEvolutionController
from cineos.native_image.evolution_search import EvolutionConfig
from cineos.native_image.evolution_state import EvolutionResumeState, EvolutionStateStore
from cineos.native_image.model_tournament import TournamentCandidate
from cineos.native_image.training_budget import (
    ExperimentBudgetController,
    ResourceUsage,
    TrainingBudget,
)


def _trainer(config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.pt"
    checkpoint.write_text(config.candidate_id)
    score = CheckpointScore(config.candidate_id, 0.2, 0.4, 0.5, 0.8)
    return TournamentCandidate(config.candidate_id, str(checkpoint), score), 0.25


def test_budget_runtime_records_candidate_and_gpu_usage(tmp_path):
    store = EvolutionStateStore(tmp_path / "resume.json")
    runtime = BudgetAwareEvolutionRuntime(
        MultiGenerationEvolutionController(CheckpointBenchmarkGate(), 2),
        ExperimentBudgetController(TrainingBudget(max_generations=2, max_gpu_hours=2.0)),
    )
    seed = EvolutionConfig(1e-3, 16, 64, 8, 1.0, 1.0)
    result = runtime.run(seed, _trainer, tmp_path / "work", store, run_id="budget-1")
    assert result.usage.generations_completed == 1
    assert result.usage.candidates_trained == 2
    assert result.usage.gpu_hours_used == 0.5
    assert store.load().usage == result.usage


def test_budget_runtime_refuses_exhausted_resume_state(tmp_path):
    store = EvolutionStateStore(tmp_path / "resume.json")
    store.save(
        EvolutionResumeState(
            run_id="budget-2",
            current_config=EvolutionConfig(1e-3, 16, 64, 8, 1.0, 1.0),
            usage=ResourceUsage(generations_completed=1),
        )
    )
    runtime = BudgetAwareEvolutionRuntime(
        MultiGenerationEvolutionController(CheckpointBenchmarkGate()),
        ExperimentBudgetController(TrainingBudget(max_generations=1)),
    )
    result = runtime.run(
        EvolutionConfig(1e-3, 16, 64, 8, 1.0, 1.0),
        _trainer,
        tmp_path / "work",
        store,
        run_id="budget-2",
    )
    assert result.evolution is None
    assert result.stopped_reason == "generation budget exhausted"
