"""Versioned observability events for CINEOS native temporal generation.

The native renderer must remain auditable without coupling core generation to a
specific telemetry vendor. This module therefore defines a small, stable event
contract plus standard-library sinks. Events are immutable and schema-versioned
so future analytics and migration tooling can distinguish historical formats.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Protocol

TEMPORAL_EVENT_SCHEMA = "cineos.native_video.temporal_event.v1"


@dataclass(frozen=True, slots=True)
class TemporalRuntimeEvent:
    """Immutable audit event emitted by the transactional temporal runtime."""

    event_type: str
    shot_id: str
    frame_index: int
    attempt: int
    decision: str
    continuity_delta: float
    threshold: float
    schema_version: str = TEMPORAL_EVENT_SCHEMA
    occurred_at: str = ""

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("event_type cannot be empty")
        if not self.shot_id:
            raise ValueError("shot_id cannot be empty")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.attempt <= 0:
            raise ValueError("attempt must be positive")
        if self.continuity_delta < 0:
            raise ValueError("continuity_delta must be non-negative")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        if not self.occurred_at:
            object.__setattr__(
                self,
                "occurred_at",
                datetime.now(UTC).isoformat(),
            )

    def to_record(self) -> dict[str, object]:
        """Return a JSON-serializable record with a stable field contract."""
        return asdict(self)


class TemporalObserver(Protocol):
    """Consumer for temporal runtime events."""

    def record(self, event: TemporalRuntimeEvent) -> None: ...


class NullTemporalObserver:
    """Default no-op observer used when telemetry is not configured."""

    def record(self, event: TemporalRuntimeEvent) -> None:
        del event


class InMemoryTemporalObserver:
    """Thread-safe event collector useful for tests and embedded runtimes."""

    def __init__(self) -> None:
        self._events: list[TemporalRuntimeEvent] = []
        self._lock = Lock()

    def record(self, event: TemporalRuntimeEvent) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> tuple[TemporalRuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)


class JsonlTemporalObserver:
    """Append temporal events to a durable JSONL audit log.

    The file is opened per write to avoid retaining stale descriptors across
    worker restarts. A process-local lock keeps records intact for concurrent
    threads. Multi-process aggregation should use separate worker files or a
    future external telemetry adapter implementing :class:`TemporalObserver`.
    """

    def __init__(self, path: str | Path, *, fsync: bool = False) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self._lock = Lock()

    def record(self, event: TemporalRuntimeEvent) -> None:
        payload = json.dumps(event.to_record(), sort_keys=True, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                if self.fsync:
                    import os

                    os.fsync(handle.fileno())
