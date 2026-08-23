"""Production-style training job orchestration primitives for CINEOS."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


JOB_STATES = {"queued", "running", "completed", "failed", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")


@dataclass(frozen=True, slots=True)
class TrainingJob:
    job_id: str
    state: str = "queued"
    attempts: int = 0
    heartbeat_at: str | None = None
    checkpoint_path: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must not be empty")
        if self.state not in JOB_STATES:
            raise ValueError("unsupported training job state")


class TrainingJobStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, job: TrainingJob) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(job), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.path)
        return self.path

    def load(self) -> TrainingJob:
        return TrainingJob(**json.loads(self.path.read_text(encoding="utf-8")))


JobWorker = Callable[[TrainingJob], str]


@dataclass(slots=True)
class TrainingJobOrchestrator:
    store: TrainingJobStore
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    def submit(self, job_id: str) -> TrainingJob:
        job = TrainingJob(job_id=job_id)
        self.store.save(job)
        return job

    def heartbeat(self, job: TrainingJob) -> TrainingJob:
        updated = replace(job, heartbeat_at=_now(), updated_at=_now())
        self.store.save(updated)
        return updated

    def cancel(self, job: TrainingJob) -> TrainingJob:
        if job.state in {"completed", "cancelled"}:
            return job
        updated = replace(job, state="cancelled", updated_at=_now())
        self.store.save(updated)
        return updated

    def run(self, job: TrainingJob, worker: JobWorker) -> TrainingJob:
        current = job
        while current.attempts < self.retry_policy.max_attempts:
            if current.state == "cancelled":
                return current
            current = replace(
                current,
                state="running",
                attempts=current.attempts + 1,
                heartbeat_at=_now(),
                error=None,
                updated_at=_now(),
            )
            self.store.save(current)
            try:
                checkpoint_path = worker(current)
            except Exception as exc:
                current = replace(
                    current,
                    state="failed",
                    error=str(exc),
                    updated_at=_now(),
                )
                self.store.save(current)
                if current.attempts >= self.retry_policy.max_attempts:
                    return current
                current = replace(current, state="queued", updated_at=_now())
                self.store.save(current)
                continue
            current = replace(
                current,
                state="completed",
                checkpoint_path=checkpoint_path,
                updated_at=_now(),
            )
            self.store.save(current)
            return current
        return current
