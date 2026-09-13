import pytest

from cineos.native_image.conditional_eval import identity_consistency_score
from cineos.native_image.neural_backend import _load_torch, torch_available


def test_identity_consistency_requires_multiple_samples():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = _load_torch()
    with pytest.raises(ValueError, match="at least two"):
        identity_consistency_score((torch.zeros(4),))


def test_identity_consistency_is_high_for_identical_latents():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = _load_torch()
    latent = torch.tensor([0.2, -0.4, 0.8, 0.1])
    score = identity_consistency_score((latent, latent.clone(), latent.clone()))
    assert score == pytest.approx(1.0)


def test_identity_consistency_detects_large_direction_change():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = _load_torch()
    first = torch.tensor([1.0, 0.0, 0.0, 0.0])
    second = torch.tensor([0.0, 1.0, 0.0, 0.0])
    score = identity_consistency_score((first, second))
    assert score == pytest.approx(0.0)
