"""Execution engine for generic runtime tasks."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .context import RuntimeContext
from .job import RenderJob


class TaskExecutor:
    """Thin, lifecycle-managed wrapper around a thread pool."""

    def __init__(self, max_workers: int = 1) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="cineos-atlas"
        )

    def submit(self, job: RenderJob, context: RuntimeContext) -> Future[Any]:
        return self._pool.submit(job.task, context)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)

    def __enter__(self) -> TaskExecutor:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
