from dataclasses import replace
from datetime import UTC, datetime, timedelta

from cineos.native_image.gpu_scheduler import (
    GPUJobRequirements,
    GPUJobScheduler,
    GPUWorker,
    GPUWorkerPool,
)
from cineos.native_image.scheduled_leases import ScheduledLeaseRuntime
from cineos.native_image.training_jobs import TrainingJobOrchestrator, TrainingJobStore
from cineos.native_image.worker_lease import WorkerLeaseManager


def _runtime(tmp_path):
    pool = GPUWorkerPool()
    pool.register(GPUWorker("gpu-a", "A100", 80, current_load=0.1))
    pool.register(GPUWorker("gpu-b", "A100", 80, current_load=0.2))
    store = TrainingJobStore(tmp_path / "job.json")
    return (
        ScheduledLeaseRuntime(
            GPUJobScheduler(pool), pool, WorkerLeaseManager(store, timeout_seconds=30)
        ),
        TrainingJobOrchestrator(store),
    )


def test_dispatch_reserves_selected_gpu(tmp_path):
    runtime, jobs = _runtime(tmp_path)
    scheduled = runtime.dispatch(jobs.submit("job-1"), GPUJobRequirements(40))
    assert scheduled.worker_id == "gpu-a"
    workers = {worker.worker_id: worker for worker in runtime.pool.workers()}
    assert workers["gpu-a"].available is False


def test_second_job_uses_other_gpu_when_first_is_reserved(tmp_path):
    runtime, jobs = _runtime(tmp_path)
    runtime.dispatch(jobs.submit("job-1"), GPUJobRequirements(40))
    second = jobs.submit("job-2")
    scheduled = runtime.dispatch(second, GPUJobRequirements(40))
    assert scheduled.worker_id == "gpu-b"


def test_stale_worker_releases_capacity_and_job_is_rescheduled(tmp_path):
    runtime, jobs = _runtime(tmp_path)
    scheduled = runtime.dispatch(jobs.submit("job-1"), GPUJobRequirements(40))
    old = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    stale_job = replace(scheduled.job, heartbeat_at=old)
    stale = replace(scheduled, job=stale_job)
    runtime.pool.set_available("gpu-a", False)
    rescheduled = runtime.recover_and_reschedule(stale, GPUJobRequirements(40))
    assert rescheduled is not None
    assert rescheduled.job.state == "running"
    assert rescheduled.worker_id in {"gpu-a", "gpu-b"}


def test_release_returns_gpu_to_pool(tmp_path):
    runtime, jobs = _runtime(tmp_path)
    scheduled = runtime.dispatch(jobs.submit("job-1"), GPUJobRequirements(40))
    runtime.release(scheduled)
    workers = {worker.worker_id: worker for worker in runtime.pool.workers()}
    assert workers[scheduled.worker_id].available is True
