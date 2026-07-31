"""Public API for the generic CINEOS plugin framework."""

from .errors import (
    PluginCompatibilityError,
    PluginDependencyError,
    PluginError,
    PluginNotFoundError,
    PluginValidationError,
)
from .manager import ENTRY_POINT_GROUP, PLUGIN_API_VERSION, PluginManager
from .metadata import PluginMetadata
from .plugin import Plugin

__all__ = [
    "ENTRY_POINT_GROUP",
    "PLUGIN_API_VERSION",
    "Plugin",
    "PluginCompatibilityError",
    "PluginDependencyError",
    "PluginError",
    "PluginManager",
    "PluginMetadata",
    "PluginNotFoundError",
    "PluginValidationError",
]
