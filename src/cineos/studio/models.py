"""Presentation-only models; domain behavior remains in CINEOS subsystems."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class QueueState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class RenderQueueItem:
    job_id: str
    shot_id: str
    renderer: str
    state: QueueState = QueueState.QUEUED
    progress: float = 0.0
    attempts: int = 0
    started_at: str | None = None
    duration: float = 0.0
    output: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.progress <= 1:
            raise ValueError("progress must be between zero and one")

    def start(self) -> None:
        self.state = QueueState.RUNNING
        self.attempts += 1
        self.started_at = datetime.now(UTC).isoformat()


@dataclass(slots=True)
class ReviewResult:
    shot_id: str
    approved: bool = False
    manual_review_required: bool = False
    scores: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    recovery_history: list[dict[str, Any]] = field(default_factory=list)
