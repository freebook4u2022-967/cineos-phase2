"""Synchronous, thread-safe runtime event dispatch."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class RuntimeEvent:
    name: str
    job_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


EventCallback = Callable[[RuntimeEvent], None]


class EventBus:
    """Publish events to callbacks without holding locks during user code."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._callbacks: dict[str, list[EventCallback]] = defaultdict(list)
        self._lock = RLock()
        self._logger = logger or logging.getLogger(__name__)

    def subscribe(self, event_name: str, callback: EventCallback) -> Callable[[], None]:
        with self._lock:
            self._callbacks[event_name].append(callback)

        def unsubscribe() -> None:
            self.unsubscribe(event_name, callback)

        return unsubscribe

    def unsubscribe(self, event_name: str, callback: EventCallback) -> None:
        with self._lock:
            callbacks = self._callbacks.get(event_name, [])
            if callback in callbacks:
                callbacks.remove(callback)

    def emit(self, event: RuntimeEvent) -> None:
        with self._lock:
            callbacks = [
                *self._callbacks.get(event.name, ()),
                *self._callbacks.get("*", ()),
            ]
        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                self._logger.exception("runtime event callback failed: %s", event.name)
