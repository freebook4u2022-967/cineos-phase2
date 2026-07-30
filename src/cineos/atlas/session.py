"""High-level renderer session orchestration."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

from .adapter import RendererAdapter, RendererState
from .base_renderer import BaseRenderer
from .capabilities import NegotiatedCapabilities, RendererCapabilities, Resolution


class RendererSession:
    """Own a renderer lifecycle and its negotiated configuration."""

    def __init__(self, renderer: BaseRenderer | RendererAdapter) -> None:
        self._adapter = (
            renderer
            if isinstance(renderer, RendererAdapter)
            else RendererAdapter(renderer)
        )
        self._negotiated: NegotiatedCapabilities | None = None

    @property
    def capabilities(self) -> RendererCapabilities:
        return self._adapter.capabilities

    @property
    def negotiated(self) -> NegotiatedCapabilities | None:
        return self._negotiated

    @property
    def state(self) -> RendererState:
        return self._adapter.state

    def start(self, model: str | None = None, **model_options: Any) -> None:
        """Initialize, load, and warm up the renderer."""

        self._adapter.initialize()
        self._adapter.load_model(model, **model_options)
        self._adapter.warmup()

    def negotiate(
        self,
        *,
        resolution: Resolution | tuple[int, int],
        duration: float,
        fps: float,
        features: tuple[str, ...] = (),
    ) -> NegotiatedCapabilities:
        self._negotiated = self.capabilities.negotiate(
            resolution=resolution, duration=duration, fps=fps, features=features
        )
        return self._negotiated

    def render(self, request: Any) -> Any:
        return self._adapter.render(request)

    def close(self) -> None:
        self._adapter.shutdown()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
