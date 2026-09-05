from cineos.native_image.checkpoint_gate import CheckpointBenchmarkGate, CheckpointScore
from cineos.native_image.evolution_controller import MultiGenerationEvolutionController
from cineos.native_image.evolution_search import EvolutionConfig
from cineos.native_image.model_tournament import TournamentCandidate


def _trainer(config, directory):
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


def test_multi_generation_controller_runs_and_keeps_best(tmp_path):
    seed = EvolutionConfig(1e-3, 16, 64, 8, 1.0, 1.0)
    result = MultiGenerationEvolutionController(
        CheckpointBenchmarkGate(), candidates_per_generation=3
    ).run(seed, _trainer, tmp_path, generations=3)

    assert len(result.generations) == 3
    assert result.best_score.quality_score > 0
    assert any(item.promoted for item in result.generations)


def test_multi_generation_controller_requires_generations(tmp_path):
    seed = EvolutionConfig(1e-3, 16, 64, 8, 1.0, 1.0)
    controller = MultiGenerationEvolutionController(CheckpointBenchmarkGate())
    try:
        controller.run(seed, _trainer, tmp_path, generations=0)
    except ValueError as exc:
        assert "generations" in str(exc)
    else:
        raise AssertionError("expected generations validation")
