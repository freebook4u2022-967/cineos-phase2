from cineos.native_image.gpu_scheduler import (
    GPUJobRequirements,
    GPUJobScheduler,
    GPUWorker,
    GPUWorkerPool,
)


def _pool():
    pool = GPUWorkerPool()
    pool.register(GPUWorker("a", "A100", 80, current_load=0.7))
    pool.register(GPUWorker("b", "A100", 40, current_load=0.2))
    pool.register(GPUWorker("c", "H100", 80, current_load=0.4))
    return pool


def test_scheduler_selects_lowest_load_eligible_worker():
    decision = GPUJobScheduler(_pool()).select(GPUJobRequirements(minimum_vram_gb=32))
    assert decision.worker.worker_id == "b"


def test_scheduler_honors_preferred_gpu_type_when_eligible():
    decision = GPUJobScheduler(_pool()).select(
        GPUJobRequirements(minimum_vram_gb=40, preferred_gpu_type="H100")
    )
    assert decision.worker.worker_id == "c"
    assert decision.reason == "preferred GPU selected"


def test_scheduler_rejects_workers_without_required_vram():
    decision = GPUJobScheduler(_pool()).select(GPUJobRequirements(minimum_vram_gb=96))
    assert decision.worker is None
    assert decision.reason == "no eligible GPU worker"


def test_unavailable_worker_is_not_scheduled():
    pool = _pool()
    pool.set_available("b", False)
    decision = GPUJobScheduler(pool).select(GPUJobRequirements(minimum_vram_gb=32))
    assert decision.worker.worker_id == "c"


def test_worker_load_updates_change_scheduling_decision():
    pool = _pool()
    pool.update_load("b", 0.9)
    pool.update_load("a", 0.1)
    decision = GPUJobScheduler(pool).select(GPUJobRequirements(minimum_vram_gb=32))
    assert decision.worker.worker_id == "a"
