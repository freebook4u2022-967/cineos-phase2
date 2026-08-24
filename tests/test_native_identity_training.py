import pytest

from cineos.native_image.identity_loss import IdentityLossConfig
from cineos.native_image.identity_training import (
    IdentityAwareTrainingStep,
    TorchIdentityProjection,
)
from cineos.native_image.neural_backend import _load_torch, torch_available


def test_identity_projection_validates_dimensions():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    with pytest.raises(ValueError, match="positive"):
        TorchIdentityProjection(0, 4)


def test_identity_aware_step_updates_projection_parameters():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = _load_torch()
    model = torch.nn.Sequential(
        torch.nn.Linear(4 + 4 + 6 + 1, 16),
        torch.nn.SiLU(),
        torch.nn.Linear(16, 6),
    )

    class Wrapped:
        def __call__(self, identity, scene, latent, time):
            return model(torch.cat((identity, scene, latent, time), dim=-1))

    projection = TorchIdentityProjection(6, 4)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projection.module.parameters()), lr=1e-2
    )
    step = IdentityAwareTrainingStep(
        Wrapped(),
        projection,
        optimizer,
        identity_loss_config=IdentityLossConfig(weight=0.5),
    )
    before = projection.module.weight.detach().clone()
    batch = 3
    result = step.train_batch(
        torch.randn(batch, 4),
        torch.randn(batch, 4),
        torch.zeros(batch, 6),
        torch.randn(batch, 6),
        torch.nn.functional.normalize(torch.randn(batch, 4), dim=-1),
    )
    assert result.total_loss > 0
    assert result.flow_loss >= 0
    assert result.identity_loss >= 0
    assert not torch.equal(before, projection.module.weight.detach())
