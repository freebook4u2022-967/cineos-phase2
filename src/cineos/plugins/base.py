"""Base contract implemented by CINEOS plugins."""

from abc import ABC

from .metadata import PluginMetadata


class Plugin(ABC):
    """Base plugin with overridable lifecycle hooks.

    Hooks intentionally default to no-ops so a plugin only implements the
    lifecycle work it needs. The manager, rather than plugin code, owns state.
    """

    metadata: PluginMetadata

    def initialize(self) -> None:
        """Prepare resources after the plugin is loaded."""

    def start(self) -> None:
        """Activate the initialized plugin."""

    def stop(self) -> None:
        """Deactivate the plugin while retaining initialized resources."""

    def shutdown(self) -> None:
        """Release resources before the plugin is unloaded."""
