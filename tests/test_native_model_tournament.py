from cineos.native_image.checkpoint_gate import CheckpointBenchmarkGate, CheckpointScore
from cineos.native_image.model_tournament import AutomatedModelTournament, TournamentCandidate


def _trainer(candidate_id, directory):
    quality = {"weak": 0.3, "strong": 0.9, "middle": 0.6}[candidate_id]
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.pt"
    checkpoint.write_text(candidate_id)
    score = CheckpointScore(
        checkpoint_id=candidate_id,
        reconstruction_mse=1.0 - quality,
        same_character_scene_distance=quality * 0.5,
        different_character_same_scene_distance=quality * 0.6,
        identity_consistency_score=quality,
    )
    return TournamentCandidate(candidate_id, str(checkpoint), score)


def test_tournament_ranks_candidates_and_promotes_winner(tmp_path):
    result = AutomatedModelTournament(CheckpointBenchmarkGate()).run(
        ("weak", "strong", "middle"), _trainer, tmp_path
    )
    assert result.winner.candidate_id == "strong"
    assert [item.candidate_id for item in result.ranked] == ["strong", "middle", "weak"]
    assert result.promoted is True
    assert (tmp_path / "best_checkpoint.json").exists()


def test_tournament_rejects_winner_when_incumbent_is_better(tmp_path):
    incumbent = CheckpointScore(
        checkpoint_id="incumbent",
        reconstruction_mse=0.01,
        same_character_scene_distance=0.9,
        different_character_same_scene_distance=0.9,
        identity_consistency_score=0.99,
    )
    result = AutomatedModelTournament(CheckpointBenchmarkGate()).run(
        ("weak", "middle"), _trainer, tmp_path, incumbent=incumbent
    )
    assert result.winner.candidate_id == "middle"
    assert result.promoted is False
