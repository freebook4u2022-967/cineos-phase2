import pytest

from cineos.native_image.checkpoint_gate import CheckpointBenchmarkGate, CheckpointScore
from cineos.native_image.evolution_controller import MultiGenerationEvolutionController
from cineos.native_image.evolution_search import EvolutionConfig
from cineos.native_image.evolution_state import EvolutionStateStore
from cineos.native_image.model_tournament import TournamentCandidate


def _candidate(config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.pt"
    checkpoint.write_text(config.candidate_id)
    quality = min(0.99, 0.45 + (config.hidden_dim / 1000.0))
    score = CheckpointScore(
        checkpoint_id=config.candidate_id,
        reconstruction_mse=max(0.01, 1.0 - quality),
        same_character_scene_distance=quality * 0.4,
        different_character_same_scene_distance=quality * 0.5,
        identity_consistency_score=quality,
    )
    return TournamentCandidate(config.candidate_id, str(checkpoint), score)


def test_resumable_run_completes_and_persists_safe_generation_state(tmp_path):
    seed = EvolutionConfig(1e-3, 16, 64, 8, 1.0, 1.0)
    store = EvolutionStateStore(tmp_path / "resume.json")
    controller = MultiGenerationEvolutionController(
        CheckpointBenchmarkGate(), candidates_per_generation=2
    )

    result = controller.run_resumable(
        seed,
        _candidate,
        tmp_path / "work",
        store,
        run_id="run-safe",
        generations=2,
    )
    restored = store.load()

    assert len(result.generations) == 2
    assert restored.current_generation == 3
    assert restored.completed_generations == (1, 2)
    assert restored.candidates == ()
    assert restored.best_score == result.best_score


def test_resume_skips_completed_candidate_after_interruption(tmp_path):
    seed = EvolutionConfig(1e-3, 16, 64, 8, 1.0, 1.0)
    store = EvolutionStateStore(tmp_path / "resume.json")
    controller = MultiGenerationEvolutionController(
        CheckpointBenchmarkGate(), candidates_per_generation=2
    )
    calls = []
    crashed = {"done": False}

    def flaky_trainer(config, directory):
        calls.append(config.candidate_id)
        if len(calls) == 2 and not crashed["done"]:
            crashed["done"] = True
            raise RuntimeError("simulated worker crash")
        return _candidate(config, directory)

    with pytest.raises(RuntimeError, match="simulated worker crash"):
        controller.run_resumable(
            seed,
            flaky_trainer,
            tmp_path / "work",
            store,
            run_id="run-recovery",
            generations=1,
        )

    state_after_crash = store.load()
    completed_ids = {
        item.candidate_id
        for item in state_after_crash.candidates
        if item.status == "completed"
    }
    assert len(completed_ids) == 1
    first_completed = next(iter(completed_ids))
    first_call_count = calls.count(first_completed)

    result = controller.run_resumable(
        seed,
        flaky_trainer,
        tmp_path / "work",
        store,
        run_id="run-recovery",
        generations=1,
    )

    assert len(result.generations) == 1
    assert calls.count(first_completed) == first_call_count
    assert store.load().completed_generations == (1,)


def test_resume_rejects_wrong_run_id(tmp_path):
    seed = EvolutionConfig(1e-3, 16, 64, 8, 1.0, 1.0)
    store = EvolutionStateStore(tmp_path / "resume.json")
    controller = MultiGenerationEvolutionController(
        CheckpointBenchmarkGate(), candidates_per_generation=1
    )
    controller.run_resumable(
        seed,
        _candidate,
        tmp_path / "work",
        store,
        run_id="original",
        generations=1,
    )

    with pytest.raises(ValueError, match="different evolution run"):
        controller.run_resumable(
            seed,
            _candidate,
            tmp_path / "work",
            store,
            run_id="other",
            generations=1,
        )
