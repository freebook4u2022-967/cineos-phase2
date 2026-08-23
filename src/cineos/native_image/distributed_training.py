"""Distributed training coordination primitives for CINEOS."""

from __future__ import annotations

from dataclasses import dataclass

from .gpu_scheduler import GPUWorker


@dataclass(frozen=True, slots=True)
class DistributedTrainingPlan:
    workers: tuple[GPUWorker, ...]
    world_size: int
    shard_count: int


@dataclass(frozen=True, slots=True)
class DataShard:
    rank: int
    start_index: int
    end_index: int


class DistributedTrainingCoordinator:
    def create_plan(
        self,
        workers: tuple[GPUWorker, ...],
        *,
        minimum_vram_gb: float,
        requested_world_size: int,
    ) -> DistributedTrainingPlan:
        if requested_world_size < 1:
            raise ValueError("requested_world_size must be positive")
        eligible = tuple(
            worker
            for worker in workers
            if worker.available and worker.vram_gb >= minimum_vram_gb
        )
        if len(eligible) < requested_world_size:
            raise RuntimeError("insufficient eligible GPU workers for distributed training")
        selected = tuple(
            sorted(eligible, key=lambda item: (item.current_load, -item.vram_gb, item.worker_id))[
                :requested_world_size
            ]
        )
        return DistributedTrainingPlan(selected, requested_world_size, requested_world_size)

    def shard_dataset(self, sample_count: int, world_size: int) -> tuple[DataShard, ...]:
        if sample_count < 1 or world_size < 1:
            raise ValueError("sample_count and world_size must be positive")
        base, remainder = divmod(sample_count, world_size)
        shards = []
        cursor = 0
        for rank in range(world_size):
            size = base + (1 if rank < remainder else 0)
            end = cursor + size
            shards.append(DataShard(rank, cursor, end))
            cursor = end
        return tuple(shards)

    def average_gradients(self, gradients: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
        if not gradients:
            raise ValueError("at least one gradient vector is required")
        width = len(gradients[0])
        if width == 0 or any(len(vector) != width for vector in gradients):
            raise ValueError("gradient vectors must have equal non-zero dimensions")
        count = float(len(gradients))
        return tuple(sum(vector[index] for vector in gradients) / count for index in range(width))
