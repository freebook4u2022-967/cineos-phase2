from cineos.native_image.checkpoint_gate import CheckpointBenchmarkGate, CheckpointScore


def _score(checkpoint_id: str, *, identity: float, scene: float, mse: float):
    return CheckpointScore(
        checkpoint_id=checkpoint_id,
        reconstruction_mse=mse,
        same_character_scene_distance=scene,
        different_character_same_scene_distance=scene,
        identity_consistency_score=identity,
    )


def test_first_checkpoint_is_promoted(tmp_path):
    candidate = _score("v1", identity=0.7, scene=0.3, mse=0.4)
    registry = tmp_path / "best.json"
    decision = CheckpointBenchmarkGate().promote_if_better(candidate, None, registry)
    assert decision.promoted is True
    assert registry.exists()


def test_better_checkpoint_replaces_incumbent(tmp_path):
    incumbent = _score("v1", identity=0.5, scene=0.2, mse=0.6)
    candidate = _score("v2", identity=0.8, scene=0.4, mse=0.3)
    registry = tmp_path / "best.json"
    decision = CheckpointBenchmarkGate().promote_if_better(
        candidate, incumbent, registry
    )
    assert decision.promoted is True
    assert '"checkpoint_id": "v2"' in registry.read_text()


def test_worse_checkpoint_is_rejected_without_overwriting_registry(tmp_path):
    incumbent = _score("v1", identity=0.9, scene=0.5, mse=0.2)
    candidate = _score("v2", identity=0.4, scene=0.1, mse=0.8)
    registry = tmp_path / "best.json"
    registry.write_text("keep-me")
    decision = CheckpointBenchmarkGate().promote_if_better(
        candidate, incumbent, registry
    )
    assert decision.promoted is False
    assert registry.read_text() == "keep-me"
