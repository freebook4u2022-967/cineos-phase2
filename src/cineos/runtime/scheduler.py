"""Scheduler coordinating queued jobs and task execution."""

from __future__ import annotations

import logging
from concurrent.futures import Future
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from .context import JobCancelledError, RuntimeContext
from .events import EventBus, RuntimeEvent
from .executor import TaskExecutor
from .job import JobState, RenderJob
from .queue import RenderQueue


class Scheduler:
    """Dispatch jobs, track lifecycle state, and implement retries."""

    def __init__(
        self,
        queue: RenderQueue,
        executor: TaskExecutor,
        events: EventBus,
        logger: logging.Logger | None = None,
    ) -> None:
        self.queue = queue
        self.executor = executor
        self.events = events
        self.logger = logger or logging.getLogger(__name__)
        self._contexts: dict[str, RuntimeContext] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._lock = RLock()

    def dispatch_next(self, timeout: float | None = None) -> Future[Any]:
        job = self.queue.get(timeout=timeout)
        with job._lock:
            job.state = JobState.RUNNING
            job.attempts += 1
            job.started_at = datetime.now(UTC)
        context = RuntimeContext(
            job.id, _progress_callback=lambda value: self._progress(job, value)
        )
        with self._lock:
            self._contexts[job.id] = context
        self.logger.info("starting job %s (attempt %d)", job.id, job.attempts)
        self.events.emit(RuntimeEvent("job.started", job.id))
        future = self.executor.submit(job, context)
        future.add_done_callback(lambda completed: self._complete(job, completed))
        with self._lock:
            self._futures[job.id] = future
        return future

    def cancel(self, job_id: str) -> bool:
        job = self.queue.get_job(job_id)
        if job is None or job.terminal:
            return False
        with self._lock:
            context = self._contexts.get(job_id)
            future = self._futures.get(job_id)
        if context is not None:
            context.cancel()
        if future is not None:
            future.cancel()
        with job._lock:
            job.state = JobState.CANCELLED
        self.events.emit(RuntimeEvent("job.cancelled", job.id))
        self.logger.info("cancelled job %s", job.id)
        return True

    def retry(self, job_id: str) -> bool:
        job = self.queue.get_job(job_id)
        if job is None or job.state != JobState.FAILED:
            return False
        with job._lock:
            job.state = JobState.QUEUED
            job.error = None
            job.finished_at = None
        # Queue identity remains registered, so retry through the internal FIFO.
        self.queue.requeue(job)
        self.events.emit(RuntimeEvent("job.retried", job.id))
        return True

    def _progress(self, job: RenderJob, value: float) -> None:
        job.update_progress(value)
        self.events.emit(RuntimeEvent("job.progress", job.id, {"progress": value}))

    def _complete(self, job: RenderJob, future: Future[Any]) -> None:
        try:
            result = future.result()
        except JobCancelledError as error:
            with job._lock:
                job.error = error
                job.state = JobState.CANCELLED
            event_name = "job.cancelled"
        except BaseException as error:
            with job._lock:
                job.error = error
                job.state = JobState.FAILED
            self.logger.exception(
                "job %s failed",
                job.id,
                exc_info=(type(error), error, error.__traceback__),
            )
            if job.attempts <= job.max_retries:
                self.retry(job.id)
                event_name = "job.retrying"
            else:
                event_name = "job.failed"
        else:
            with job._lock:
                if job.state != JobState.CANCELLED:
                    job.result = result
                    job.progress = 1.0
                    job.state = JobState.COMPLETED
            event_name = (
                "job.cancelled" if job.state == JobState.CANCELLED else "job.completed"
            )
        with job._lock:
            if job.state != JobState.QUEUED:
                job.finished_at = datetime.now(UTC)
        self.queue.task_done()
        self.events.emit(RuntimeEvent(event_name, job.id))

    def context_for(self, job_id: str) -> RuntimeContext | None:
        with self._lock:
            return self._contexts.get(job_id)
