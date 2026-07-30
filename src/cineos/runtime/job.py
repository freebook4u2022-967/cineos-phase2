"""Render job value types and lifecycle state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import Any
from uuid import uuid4


class JobState(StrEnum):
    """States in the lifecycle of a render job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


JobTask = Callable[["RuntimeContext"], Any]


@dataclass
class RenderJob:
    """A unit of runtime work.

    The task is intentionally a generic callable. The Atlas runtime coordinates
    work but contains no rendering, GPU, or AI implementation.
    """

    task: JobTask
    name: str = "render-job"
    max_retries: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    state: JobState = field(default=JobState.QUEUED, init=False)
    progress: float = field(default=0.0, init=False)
    attempts: int = field(default=0, init=False)
    result: Any = field(default=None, init=False)
    error: BaseException | None = field(default=None, init=False, repr=False)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC), init=False)
    started_at: datetime | None = field(default=None, init=False)
    finished_at: datetime | None = field(default=None, init=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not callable(self.task):
            raise TypeError("task must be callable")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

    @property
    def terminal(self) -> bool:
        return self.state in {
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        }

    def update_progress(self, progress: float) -> None:
        """Set completion progress, expressed as a number from 0 through 1."""
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be between 0 and 1")
        with self._lock:
            self.progress = float(progress)


from .context import RuntimeContext  # noqa: E402  (typing cycle)
