"""Worker lease and stale-job recovery for CINEOS training jobs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from .training_jobs import TrainingJob, TrainingJobStore, _now


@dataclass(frozen=True, slots=True)
class WorkerLease:
    worker_id: str
    acquired_at: str
    expires_at: str

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")


@dataclass(slots=True)
class WorkerLeaseManager:
    store: TrainingJobStore
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")

    def is_stale(self, job: TrainingJob, *, now: datetime | None = None) -> bool:
        if job.state != "running" or job.heartbeat_at is None:
            return False
        current = now or datetime.now(timezone.utc)
        heartbeat = datetime.fromisoformat(job.heartbeat_at)
        if heartbeat.tzinfo is None:
            heartbeat = heartbeat.replace(tzinfo=timezone.utc)
        return current - heartbeat > timedelta(seconds=self.timeout_seconds)

    def recover_stale(self, job: TrainingJob) -> TrainingJob:
        if not self.is_stale(job):
            return job
        recovered = replace(
            job,
            state="queued",
            heartbeat_at=None,
            error="stale worker lease recovered",
            updated_at=_now(),
        )
        self.store.save(recovered)
        return recovered

    def acquire(self, job: TrainingJob, worker_id: str) -> tuple[TrainingJob, WorkerLease]:
        if job.state not in {"queued", "failed"}:
            raise ValueError("only queued or failed jobs may acquire a worker lease")
        acquired = datetime.now(timezone.utc)
        expires = acquired + timedelta(seconds=self.timeout_seconds)
        leased = replace(
            job,
            state="running",
            heartbeat_at=acquired.isoformat(),
            error=None,
            updated_at=acquired.isoformat(),
        )
        self.store.save(leased)
        return leased, WorkerLease(worker_id, acquired.isoformat(), expires.isoformat())

    def renew(self, job: TrainingJob, lease: WorkerLease) -> tuple[TrainingJob, WorkerLease]:
        if job.state != "running":
            raise ValueError("only running jobs may renew a worker lease")
        renewed_at = datetime.now(timezone.utc)
        renewed = replace(
            job,
            heartbeat_at=renewed_at.isoformat(),
            updated_at=renewed_at.isoformat(),
        )
        renewed_lease = WorkerLease(
            lease.worker_id,
            lease.acquired_at,
            (renewed_at + timedelta(seconds=self.timeout_seconds)).isoformat(),
        )
        self.store.save(renewed)
        return renewed, renewed_lease
