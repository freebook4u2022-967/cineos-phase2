"""Abstract renderer contract for Atlas integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .capabilities import RendererCapabilities


class BaseRenderer(ABC):
    """Backend-neutral renderer lifecycle.

    Implementations own their model-specific input and output types. Atlas only
    defines when lifecycle operations occur and how capabilities are exposed.
    """

    @property
    @abstractmethod
    def capabilities(self) -> RendererCapabilities:
        """Return the renderer's static capability declaration."""

    @abstractmethod
    def initialize(self) -> None:
        """Allocate resources needed by the renderer."""

    @abstractmethod
    def load_model(self, model: str | None = None, **options: Any) -> None:
        """Load a model or backend configuration."""

    @abstractmethod
    def warmup(self) -> None:
        """Prepare the loaded renderer for its first request."""

    @abstractmethod
    def render(self, request: Any) -> Any:
        """Render a backend-defined request and return a backend-defined result."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release all resources. Implementations should make this idempotent."""
