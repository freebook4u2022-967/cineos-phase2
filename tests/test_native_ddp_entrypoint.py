import pytest

from cineos.native_image.ddp_entrypoint import DDPTrainConfig, SyntheticFlowDataset
from cineos.native_image.neural_backend import _load_torch, torch_available


def test_ddp_train_config_defaults_are_valid():
    config = DDPTrainConfig()
    assert config.samples > 0
    assert config.batch_size > 0
    assert config.steps > 0


def test_synthetic_flow_dataset_shapes():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = _load_torch()
    config = DDPTrainConfig(samples=6, feature_dim=4, latent_dim=5)
    dataset = SyntheticFlowDataset(torch, config)
    identity, scene, source, target = dataset[0]
    assert len(dataset) == 6
    assert tuple(identity.shape) == (4,)
    assert tuple(scene.shape) == (4,)
    assert tuple(source.shape) == (5,)
    assert tuple(target.shape) == (5,)
