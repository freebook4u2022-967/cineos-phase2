"""Lifecycle-safe adapter around renderer implementations."""

from __future__ import annotations

from enum import Enum, auto
from typing import Any

from .base_renderer import BaseRenderer
from .capabilities import RendererCapabilities


class RendererState(Enum):
    """Observable lifecycle states for :class:`RendererAdapter`."""

    NEW = auto()
    INITIALIZED = auto()
    MODEL_LOADED = auto()
    READY = auto()
    SHUTDOWN = auto()


class RendererLifecycleError(RuntimeError):
    """Raised when renderer lifecycle methods are called out of order."""


class RendererAdapter:
    """Enforce lifecycle ordering for a :class:`BaseRenderer`."""

    def __init__(self, renderer: BaseRenderer) -> None:
        self._renderer = renderer
        self._state = RendererState.NEW

    @property
    def renderer(self) -> BaseRenderer:
        return self._renderer

    @property
    def state(self) -> RendererState:
        return self._state

    @property
    def capabilities(self) -> RendererCapabilities:
        return self._renderer.capabilities

    def initialize(self) -> None:
        self._require(RendererState.NEW)
        self._renderer.initialize()
        self._state = RendererState.INITIALIZED

    def load_model(self, model: str | None = None, **options: Any) -> None:
        self._require(RendererState.INITIALIZED)
        self._renderer.load_model(model, **options)
        self._state = RendererState.MODEL_LOADED

    def warmup(self) -> None:
        self._require(RendererState.MODEL_LOADED)
        self._renderer.warmup()
        self._state = RendererState.READY

    def render(self, request: Any) -> Any:
        self._require(RendererState.READY)
        return self._renderer.render(request)

    def shutdown(self) -> None:
        if self._state is RendererState.SHUTDOWN:
            return
        self._renderer.shutdown()
        self._state = RendererState.SHUTDOWN

    def _require(self, expected: RendererState) -> None:
        if self._state is not expected:
            raise RendererLifecycleError(
                f"operation requires {expected.name}, renderer is {self._state.name}"
            )
