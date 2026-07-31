"""Public API for renderer-independent CINEOS plugins."""

from .errors import (
    PluginError,
    PluginLifecycleError,
    PluginLoadError,
    PluginRegistrationError,
)
from .manager import PLUGIN_ENTRY_POINT_GROUP, SUPPORTED_API_VERSION, PluginManager
from .plugin import Plugin, PluginContext, PluginMetadata

__all__ = [
    "PLUGIN_ENTRY_POINT_GROUP",
    "SUPPORTED_API_VERSION",
    "Plugin",
    "PluginContext",
    "PluginError",
    "PluginLifecycleError",
    "PluginLoadError",
    "PluginManager",
    "PluginMetadata",
    "PluginRegistrationError",
]
