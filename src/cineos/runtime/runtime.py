"""High-level Atlas runtime facade."""

from __future__ import annotations

import logging
from concurrent.futures import Future
from enum import StrEnum
from typing import Any

from .config import RuntimeConfig
from .events import EventBus, RuntimeEvent
from .executor import TaskExecutor
from .job import RenderJob
from .queue import Empty, RenderQueue
from .scheduler import Scheduler


class RuntimeState(StrEnum):
    """Observable lifecycle state of an Atlas runtime instance."""

    RUNNING = "running"
    CLOSED = "closed"


class AtlasRuntime:
    """Own and expose the Atlas runtime infrastructure."""

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self.config = config or RuntimeConfig()
        self.logger = logging.getLogger("cineos.runtime")
        self.logger.setLevel(self.config.log_level.upper())
        self.events = EventBus(self.logger)
        self.queue = RenderQueue()
        self.executor = TaskExecutor(self.config.workers)
        self.scheduler = Scheduler(self.queue, self.executor, self.events, self.logger)
        self.state = RuntimeState.RUNNING

    def submit(self, job: RenderJob) -> str:
        if self.state == RuntimeState.CLOSED:
            raise RuntimeError("runtime is closed")
        self.queue.put(job)
        self.events.emit(RuntimeEvent("job.queued", job.id))
        self.logger.info("queued job %s", job.id)
        return job.id

    def run_next(self, timeout: float | None = None) -> Future[Any]:
        if self.state == RuntimeState.CLOSED:
            raise RuntimeError("runtime is closed")
        return self.scheduler.dispatch_next(timeout)

    def run_pending(self) -> list[Future[Any]]:
        futures: list[Future[Any]] = []
        while self.queue.pending:
            try:
                futures.append(self.run_next(timeout=0))
            except Empty:
                break
        return futures

    def cancel(self, job_id: str) -> bool:
        return self.scheduler.cancel(job_id)

    def retry(self, job_id: str) -> bool:
        return self.scheduler.retry(job_id)

    def get_job(self, job_id: str) -> RenderJob | None:
        return self.queue.get_job(job_id)

    def shutdown(self, wait: bool = True) -> None:
        if self.state != RuntimeState.CLOSED:
            self.state = RuntimeState.CLOSED
            self.executor.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> AtlasRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
