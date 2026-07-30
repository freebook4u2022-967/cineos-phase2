"""Context passed to work executed by the runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event, RLock
from typing import Any


class JobCancelledError(Exception):
    """Raised when cooperative work observes a cancellation request."""


@dataclass
class RuntimeContext:
    """Thread-safe per-job state available to a job task."""

    job_id: str
    _cancelled: Event = field(default_factory=Event, repr=False)
    _progress_callback: Callable[[float], None] | None = field(default=None, repr=False)
    _values: dict[str, Any] = field(default_factory=dict, repr=False)
    _lock: RLock = field(default_factory=RLock, repr=False)

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        self._cancelled.set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelledError(f"job {self.job_id} was cancelled")

    def report_progress(self, progress: float) -> None:
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be between 0 and 1")
        self.raise_if_cancelled()
        if self._progress_callback is not None:
            self._progress_callback(float(progress))

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._values.get(key, default)
