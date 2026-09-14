"""High-level renderer session orchestration."""

from __future__ import annotations

from types import TracebackType
from typing import Any, Self

from .adapter import RendererAdapter, RendererState
from .base_renderer import BaseRenderer
from .capabilities import (
    CapabilityError,
    NegotiatedCapabilities,
    RendererCapabilities,
    Resolution,
)


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
        character_count: int | None = None,
    ) -> NegotiatedCapabilities:
        self._negotiated = self.capabilities.negotiate(
            resolution=resolution,
            duration=duration,
            fps=fps,
            features=features,
            character_count=character_count,
        )
        return self._negotiated

    @staticmethod
    def _request_character_count(request: Any) -> int | None:
        """Return the concrete cast size for request objects that expose characters."""

        if not hasattr(request, "characters"):
            return None
        characters = request.characters
        if not isinstance(characters, (list, tuple)):
            raise CapabilityError("request.characters must be a list or tuple")
        return len(characters)

    def _validate_render_character_capacity(self, request: Any) -> None:
        """Bind capability negotiation to the cast that is actually rendered.

        ``character_count`` used during negotiation is caller-supplied and therefore
        cannot be the production trust boundary. Re-derive the cast size from the
        concrete request immediately before dispatch, reject stale negotiations, and
        enforce the renderer's advertised capacity even when legacy callers omitted
        character_count during negotiation.
        """

        actual_count = self._request_character_count(request)
        if actual_count is None:
            return

        maximum = self.capabilities.maximum_character_count
        if maximum is not None and actual_count > maximum:
            raise CapabilityError(
                f"request character_count {actual_count} exceeds maximum {maximum}"
            )

        negotiated_count = (
            self._negotiated.character_count if self._negotiated is not None else None
        )
        if negotiated_count is not None and actual_count != negotiated_count:
            raise CapabilityError(
                "request character_count "
                f"{actual_count} does not match negotiated character_count {negotiated_count}"
            )

    def render(self, request: Any) -> Any:
        self._validate_render_character_capacity(request)
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
