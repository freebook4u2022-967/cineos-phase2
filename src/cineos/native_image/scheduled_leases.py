"""Integrate GPU scheduling with worker leases for CINEOS jobs."""

from __future__ import annotations

from dataclasses import dataclass

from .gpu_scheduler import GPUJobRequirements, GPUJobScheduler, GPUWorkerPool
from .training_jobs import TrainingJob
from .worker_lease import WorkerLease, WorkerLeaseManager


@dataclass(frozen=True, slots=True)
class ScheduledLease:
    job: TrainingJob
    worker_id: str
    lease: WorkerLease


@dataclass(slots=True)
class ScheduledLeaseRuntime:
    scheduler: GPUJobScheduler
    pool: GPUWorkerPool
    leases: WorkerLeaseManager

    def dispatch(
        self,
        job: TrainingJob,
        requirements: GPUJobRequirements,
    ) -> ScheduledLease | None:
        decision = self.scheduler.select(requirements)
        if decision.worker is None:
            return None
        worker = decision.worker
        leased_job, lease = self.leases.acquire(job, worker.worker_id)
        self.pool.set_available(worker.worker_id, False)
        return ScheduledLease(leased_job, worker.worker_id, lease)

    def heartbeat(self, scheduled: ScheduledLease) -> ScheduledLease:
        job, lease = self.leases.renew(scheduled.job, scheduled.lease)
        return ScheduledLease(job, scheduled.worker_id, lease)

    def release(self, scheduled: ScheduledLease) -> TrainingJob:
        self.pool.set_available(scheduled.worker_id, True)
        return scheduled.job

    def recover_and_reschedule(
        self,
        scheduled: ScheduledLease,
        requirements: GPUJobRequirements,
    ) -> ScheduledLease | None:
        recovered = self.leases.recover_stale(scheduled.job)
        if recovered.state != "queued":
            return scheduled
        self.pool.set_available(scheduled.worker_id, True)
        return self.dispatch(recovered, requirements)
