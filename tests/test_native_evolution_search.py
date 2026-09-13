import pytest

from cineos.native_image.evolution_search import (
    EvolutionConfig,
    EvolutionSearchSpace,
    mutate_config,
)


def test_search_space_generates_model_variants():
    space = EvolutionSearchSpace(
        learning_rates=(1e-3, 5e-4),
        latent_dims=(16, 32),
        hidden_dims=(64,),
        flow_steps=(8,),
        reconstruction_weights=(1.0,),
        flow_weights=(1.0,),
    )
    candidates = space.candidates()
    assert len(candidates) == 4
    assert len({item.candidate_id for item in candidates}) == 4


def test_search_space_can_limit_tournament_size():
    space = EvolutionSearchSpace(
        learning_rates=(1e-3, 5e-4),
        latent_dims=(16, 32),
        hidden_dims=(64, 128),
        flow_steps=(8,),
        reconstruction_weights=(1.0,),
        flow_weights=(1.0,),
    )
    assert len(space.candidates(max_candidates=3)) == 3


def test_winner_mutation_creates_valid_child():
    parent = EvolutionConfig(1e-3, 16, 64, 8, 1.0, 1.0)
    child = mutate_config(parent, 1)
    assert child != parent
    assert child.learning_rate > 0
    assert child.hidden_dim > parent.hidden_dim


def test_invalid_zero_loss_configuration_is_rejected():
    with pytest.raises(ValueError, match="at least one loss"):
        EvolutionConfig(1e-3, 16, 64, 8, 0.0, 0.0)
