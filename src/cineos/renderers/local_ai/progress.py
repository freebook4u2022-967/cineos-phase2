"""Atlas Runtime renderer events."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class RendererEvent:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


EventSink = Callable[[RendererEvent], None]


def null_sink(event: RendererEvent) -> None:
    del event
