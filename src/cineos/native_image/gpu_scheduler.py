"""Multi-GPU worker pool scheduling primitives for CINEOS training jobs."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class GPUWorker:
    worker_id: str
    gpu_type: str
    vram_gb: float
    current_load: float = 0.0
    available: bool = True

    def __post_init__(self) -> None:
        if not self.worker_id or not self.gpu_type:
            raise ValueError("worker_id and gpu_type must not be empty")
        if self.vram_gb <= 0:
            raise ValueError("vram_gb must be positive")
        if not 0.0 <= self.current_load <= 1.0:
            raise ValueError("current_load must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class GPUJobRequirements:
    minimum_vram_gb: float = 0.0
    preferred_gpu_type: str | None = None
    maximum_worker_load: float = 0.95

    def __post_init__(self) -> None:
        if self.minimum_vram_gb < 0:
            raise ValueError("minimum_vram_gb must be non-negative")
        if not 0.0 <= self.maximum_worker_load <= 1.0:
            raise ValueError("maximum_worker_load must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class GPUSchedulingDecision:
    worker: GPUWorker | None
    reason: str


class GPUWorkerPool:
    def __init__(self) -> None:
        self._workers: dict[str, GPUWorker] = {}

    def register(self, worker: GPUWorker) -> None:
        self._workers[worker.worker_id] = worker

    def update_load(self, worker_id: str, current_load: float) -> GPUWorker:
        worker = self._workers[worker_id]
        updated = replace(worker, current_load=current_load)
        self._workers[worker_id] = updated
        return updated

    def set_available(self, worker_id: str, available: bool) -> GPUWorker:
        worker = self._workers[worker_id]
        updated = replace(worker, available=available)
        self._workers[worker_id] = updated
        return updated

    def workers(self) -> tuple[GPUWorker, ...]:
        return tuple(self._workers.values())


@dataclass(slots=True)
class GPUJobScheduler:
    pool: GPUWorkerPool

    def select(self, requirements: GPUJobRequirements) -> GPUSchedulingDecision:
        eligible = [
            worker
            for worker in self.pool.workers()
            if worker.available
            and worker.vram_gb >= requirements.minimum_vram_gb
            and worker.current_load <= requirements.maximum_worker_load
        ]
        if not eligible:
            return GPUSchedulingDecision(None, "no eligible GPU worker")

        def rank(worker: GPUWorker) -> tuple[int, float, float, str]:
            preferred = int(
                requirements.preferred_gpu_type is not None
                and worker.gpu_type == requirements.preferred_gpu_type
            )
            return (-preferred, worker.current_load, -worker.vram_gb, worker.worker_id)

        selected = sorted(eligible, key=rank)[0]
        reason = "preferred GPU selected" if (
            requirements.preferred_gpu_type is not None
            and selected.gpu_type == requirements.preferred_gpu_type
        ) else "lowest-load eligible GPU selected"
        return GPUSchedulingDecision(selected, reason)
