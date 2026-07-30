"""Thread-safe FIFO render queue."""

from __future__ import annotations

from queue import Empty, Queue
from threading import RLock

from .job import JobState, RenderJob


class RenderQueue:
    """FIFO queue with job identity lookup and safe cancellation."""

    def __init__(self) -> None:
        self._queue: Queue[RenderJob] = Queue()
        self._jobs: dict[str, RenderJob] = {}
        self._lock = RLock()

    def put(self, job: RenderJob) -> None:
        with self._lock:
            if job.id in self._jobs:
                raise ValueError(f"job already exists: {job.id}")
            self._jobs[job.id] = job
            self._queue.put(job)

    def requeue(self, job: RenderJob) -> None:
        """Requeue an already registered job."""
        with self._lock:
            if self._jobs.get(job.id) is not job:
                raise ValueError(f"job is not registered: {job.id}")
            self._queue.put(job)

    def get(self, timeout: float | None = None) -> RenderJob:
        while True:
            job = self._queue.get(timeout=timeout)
            if job.state != JobState.CANCELLED:
                return job
            self._queue.task_done()

    def task_done(self) -> None:
        self._queue.task_done()

    def join(self) -> None:
        self._queue.join()

    def get_job(self, job_id: str) -> RenderJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.terminal:
                return False
            job.state = JobState.CANCELLED
            return True

    @property
    def pending(self) -> int:
        return self._queue.qsize()


__all__ = ["Empty", "RenderQueue"]
