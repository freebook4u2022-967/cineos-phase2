"""Generic discovery and lifecycle framework for CINEOS extensions."""

from .discovery import ENTRY_POINT_GROUP, discover_directory, discover_entry_points
from .exceptions import (
    PluginCompatibilityError,
    PluginDependencyError,
    PluginDiscoveryError,
    PluginError,
    PluginLifecycleError,
    PluginValidationError,
)
from .manager import PLUGIN_API_VERSION, PluginManager
from .metadata import PluginMetadata, parse_version, version_satisfies
from .plugin import Plugin

__all__ = [
    "ENTRY_POINT_GROUP",
    "PLUGIN_API_VERSION",
    "Plugin",
    "PluginCompatibilityError",
    "PluginDependencyError",
    "PluginDiscoveryError",
    "PluginError",
    "PluginLifecycleError",
    "PluginManager",
    "PluginMetadata",
    "PluginValidationError",
    "discover_directory",
    "discover_entry_points",
    "parse_version",
    "version_satisfies",
]
