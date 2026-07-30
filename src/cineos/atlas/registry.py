"""Renderer registration and discovery."""

from __future__ import annotations

from collections.abc import Callable

from .base_renderer import BaseRenderer

RendererFactory = Callable[[], BaseRenderer]


class RendererRegistry:
    """An in-memory registry of renderer factories."""

    def __init__(self) -> None:
        self._factories: dict[str, RendererFactory] = {}

    def register(
        self, name: str, factory: RendererFactory, *, replace: bool = False
    ) -> None:
        normalized = self._normalize(name)
        if normalized in self._factories and not replace:
            raise ValueError(f"renderer {normalized!r} is already registered")
        self._factories[normalized] = factory

    def unregister(self, name: str) -> None:
        try:
            del self._factories[self._normalize(name)]
        except KeyError as error:
            raise KeyError(f"unknown renderer {name!r}") from error

    def create(self, name: str) -> BaseRenderer:
        try:
            factory = self._factories[self._normalize(name)]
        except KeyError as error:
            raise KeyError(f"unknown renderer {name!r}") from error
        renderer = factory()
        if not isinstance(renderer, BaseRenderer):
            raise TypeError("renderer factory must return a BaseRenderer")
        return renderer

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._normalize(name) in self._factories

    @staticmethod
    def _normalize(name: str) -> str:
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("renderer name must not be empty")
        return normalized
