"""Renderer-independent plugin lifecycle contract."""

from abc import ABC
from typing import Any

from .metadata import PluginMetadata


class Plugin(ABC):
    """Base plugin with optional, synchronous lifecycle hooks."""

    metadata: PluginMetadata

    def on_load(self, context: Any) -> None:
        """Initialize resources after the plugin is registered."""

    def on_enable(self, context: Any) -> None:
        """Activate plugin behavior."""

    def on_disable(self, context: Any) -> None:
        """Deactivate plugin behavior without releasing the plugin."""

    def on_unload(self, context: Any) -> None:
        """Release resources before the plugin is removed."""
