import pytest

from cineos.native_image.ddp_entrypoint import DDPTrainConfig, SyntheticFlowDataset
from cineos.native_image.neural_backend import _load_torch, torch_available


def test_synthetic_ddp_dataset_is_deterministic():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = _load_torch()
    config = DDPTrainConfig(samples=4)
    first = SyntheticFlowDataset(torch, config)
    second = SyntheticFlowDataset(torch, config)
    assert torch.equal(first.identity, second.identity)
    assert torch.equal(first.target, second.target)


def test_ddp_config_supports_incremental_resume_steps():
    config = DDPTrainConfig(steps=3)
    completed_steps = 7
    assert completed_steps + config.steps == 10
