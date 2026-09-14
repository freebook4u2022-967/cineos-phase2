"""Optional PyTorch Distributed/DDP backend for CINEOS native image training."""

from __future__ import annotations

from dataclasses import dataclass

from .neural_backend import _load_torch


@dataclass(frozen=True, slots=True)
class DistributedRuntimeConfig:
    rank: int
    world_size: int
    backend: str = "nccl"
    init_method: str = "env://"

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be within world_size")
        if self.backend not in {"nccl", "gloo"}:
            raise ValueError("backend must be nccl or gloo")


class TorchDistributedRuntime:
    def __init__(self, config: DistributedRuntimeConfig) -> None:
        self.config = config
        self.torch = _load_torch()

    def initialize(self) -> None:
        distributed = self.torch.distributed
        if distributed.is_initialized():
            return
        distributed.init_process_group(
            backend=self.config.backend,
            init_method=self.config.init_method,
            rank=self.config.rank,
            world_size=self.config.world_size,
        )

    def wrap_model(self, model, *, device_id: int | None = None):
        ddp = self.torch.nn.parallel.DistributedDataParallel
        if device_id is None:
            return ddp(model)
        return ddp(model, device_ids=[device_id], output_device=device_id)

    def distributed_sampler(self, dataset, *, shuffle: bool = True):
        sampler = self.torch.utils.data.distributed.DistributedSampler
        return sampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=shuffle,
        )

    def barrier(self) -> None:
        self.torch.distributed.barrier()

    def is_rank_zero(self) -> bool:
        return self.config.rank == 0

    def shutdown(self) -> None:
        distributed = self.torch.distributed
        if distributed.is_initialized():
            distributed.destroy_process_group()
