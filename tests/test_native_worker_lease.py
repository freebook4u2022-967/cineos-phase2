from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from cineos.native_image.training_jobs import TrainingJobOrchestrator, TrainingJobStore
from cineos.native_image.worker_lease import WorkerLeaseManager


def test_worker_can_acquire_and_renew_job_lease(tmp_path):
    store = TrainingJobStore(tmp_path / "job.json")
    job = TrainingJobOrchestrator(store).submit("lease-1")
    manager = WorkerLeaseManager(store, timeout_seconds=60)
    running, lease = manager.acquire(job, "gpu-worker-a")
    renewed, renewed_lease = manager.renew(running, lease)
    assert running.state == "running"
    assert lease.worker_id == "gpu-worker-a"
    assert renewed.heartbeat_at is not None
    assert renewed_lease.expires_at >= lease.expires_at


def test_stale_running_job_is_requeued_for_another_worker(tmp_path):
    store = TrainingJobStore(tmp_path / "job.json")
    job = TrainingJobOrchestrator(store).submit("lease-2")
    manager = WorkerLeaseManager(store, timeout_seconds=30)
    running, _ = manager.acquire(job, "dead-worker")
    old = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    stale = replace(running, heartbeat_at=old)
    store.save(stale)
    recovered = manager.recover_stale(stale)
    assert recovered.state == "queued"
    assert recovered.heartbeat_at is None
    assert "stale worker" in recovered.error


def test_active_job_is_not_recovered(tmp_path):
    store = TrainingJobStore(tmp_path / "job.json")
    job = TrainingJobOrchestrator(store).submit("lease-3")
    manager = WorkerLeaseManager(store, timeout_seconds=300)
    running, _ = manager.acquire(job, "gpu-worker-a")
    assert manager.recover_stale(running) == running


def test_completed_job_cannot_acquire_lease(tmp_path):
    store = TrainingJobStore(tmp_path / "job.json")
    orchestrator = TrainingJobOrchestrator(store)
    completed = orchestrator.run(orchestrator.submit("lease-4"), lambda _: "model.pt")
    with pytest.raises(ValueError, match="queued or failed"):
        WorkerLeaseManager(store).acquire(completed, "gpu-worker-b")
