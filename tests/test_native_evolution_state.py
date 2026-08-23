from cineos.native_image.checkpoint_gate import CheckpointScore
from cineos.native_image.evolution_search import EvolutionConfig
from cineos.native_image.evolution_state import (
    CandidateProgress,
    EvolutionResumeState,
    EvolutionStateStore,
    recover_interrupted_candidates,
)
from cineos.native_image.training_budget import ResourceUsage


def _score(checkpoint_id: str) -> CheckpointScore:
    return CheckpointScore(checkpoint_id, 0.2, 0.4, 0.5, 0.9)


def test_evolution_state_round_trips_all_safe_resume_fields(tmp_path):
    path = tmp_path / "resume.json"
    state = EvolutionResumeState(
        run_id="run-001",
        current_generation=3,
        current_config=EvolutionConfig(1e-3, 16, 64, 8, 1.0, 1.0),
        best_score=_score("best"),
        usage=ResourceUsage(2, 6, 4.5, 1),
        candidates=(
            CandidateProgress("candidate-a", "completed", "a.pt", _score("candidate-a")),
        ),
        completed_generations=(1, 2),
    )
    store = EvolutionStateStore(path)
    store.save(state)
    restored = store.load()

    assert restored.run_id == state.run_id
    assert restored.current_generation == 3
    assert restored.current_config == state.current_config
    assert restored.best_score == state.best_score
    assert restored.usage == state.usage
    assert restored.candidates == state.candidates
    assert restored.completed_generations == (1, 2)


def test_interrupted_candidate_returns_to_pending_without_losing_checkpoint_hint():
    state = EvolutionResumeState(
        run_id="run-002",
        candidates=(CandidateProgress("candidate-a", "running", "partial.pt"),),
    )
    recovered = recover_interrupted_candidates(state)
    candidate = recovered.candidate("candidate-a")

    assert candidate.status == "pending"
    assert candidate.checkpoint_path == "partial.pt"
    assert "interrupted" in candidate.error


def test_completed_candidate_is_not_retrained_after_recovery():
    state = EvolutionResumeState(
        run_id="run-003",
        candidates=(
            CandidateProgress("done", "completed", "done.pt", _score("done")),
            CandidateProgress("next", "pending"),
        ),
    )
    recover_interrupted_candidates(state)

    assert state.candidate("done").status == "completed"
    assert state.next_pending_candidate().candidate_id == "next"
