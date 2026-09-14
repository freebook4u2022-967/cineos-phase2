import pytest

from cineos.native_image.identity_loss import (
    IdentityLossConfig,
    TorchIdentityConsistencyLoss,
    combined_training_loss,
)
from cineos.native_image.neural_backend import _load_torch, torch_available


def test_identity_loss_is_zero_for_same_embedding():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = _load_torch()
    loss_fn = TorchIdentityConsistencyLoss()
    vector = torch.tensor([[1.0, 0.0, 0.0]])
    assert float(loss_fn(vector, vector)) == pytest.approx(0.0)


def test_identity_loss_increases_for_different_identity():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = _load_torch()
    loss_fn = TorchIdentityConsistencyLoss()
    predicted = torch.tensor([[1.0, 0.0, 0.0]])
    anchor = torch.tensor([[0.0, 1.0, 0.0]])
    assert float(loss_fn(predicted, anchor)) == pytest.approx(1.0)


def test_combined_training_loss_adds_identity_term():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = _load_torch()
    total = combined_training_loss(
        torch.tensor(2.0), torch.tensor(3.0), torch.tensor(4.0)
    )
    assert float(total) == pytest.approx(9.0)


def test_invalid_identity_margin_is_rejected():
    with pytest.raises(ValueError, match="margin"):
        IdentityLossConfig(margin=1.0)
