import pytest

from cineos.native_image.neural_backend import torch_available
from cineos.native_image.torch_distributed import (
    DistributedRuntimeConfig,
    TorchDistributedRuntime,
)


def test_distributed_runtime_validates_rank():
    with pytest.raises(ValueError, match="rank"):
        DistributedRuntimeConfig(rank=2, world_size=2)


def test_distributed_runtime_validates_backend():
    with pytest.raises(ValueError, match="backend"):
        DistributedRuntimeConfig(rank=0, world_size=1, backend="invalid")


def test_rank_zero_checkpoint_authority():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    assert TorchDistributedRuntime(
        DistributedRuntimeConfig(rank=0, world_size=2, backend="gloo")
    ).is_rank_zero()
    assert not TorchDistributedRuntime(
        DistributedRuntimeConfig(rank=1, world_size=2, backend="gloo")
    ).is_rank_zero()


def test_distributed_sampler_partitions_dataset_by_rank():
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    torch = TorchDistributedRuntime(
        DistributedRuntimeConfig(rank=1, world_size=2, backend="gloo")
    ).torch
    dataset = torch.utils.data.TensorDataset(torch.arange(8))
    runtime = TorchDistributedRuntime(
        DistributedRuntimeConfig(rank=1, world_size=2, backend="gloo")
    )
    sampler = runtime.distributed_sampler(dataset, shuffle=False)
    assert list(iter(sampler)) == [1, 3, 5, 7]
