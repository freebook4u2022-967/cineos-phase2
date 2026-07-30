"""Plugin lifecycle contract."""

from __future__ import annotations

from abc import ABC
from typing import Any

from .metadata import PluginMetadata


class Plugin(ABC):
    """Base plugin with optional, renderer-independent lifecycle callbacks."""

    metadata: PluginMetadata

    def on_load(self, context: Any) -> None:
        """Run after dependencies are available and the plugin is registered."""

    def on_enable(self) -> None:
        """Run whenever the loaded plugin becomes enabled."""

    def on_disable(self) -> None:
        """Run before an enabled plugin becomes disabled."""

    def on_unload(self) -> None:
        """Run before the plugin is removed from the registry."""
