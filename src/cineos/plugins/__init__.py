"""Public plugin infrastructure for CINEOS."""

from .base import Plugin
from .exceptions import (
    DuplicatePluginError,
    PluginCompatibilityError,
    PluginDependencyError,
    PluginDiscoveryError,
    PluginError,
    PluginLifecycleError,
    PluginLoadError,
    PluginNotFoundError,
)
from .loader import PluginLoader, PluginSource
from .manager import PluginManager, PluginState
from .metadata import PluginMetadata, version_matches
from .registry import PluginRegistry

__all__ = [
    "DuplicatePluginError",
    "Plugin",
    "PluginCompatibilityError",
    "PluginDependencyError",
    "PluginDiscoveryError",
    "PluginError",
    "PluginLifecycleError",
    "PluginLoadError",
    "PluginLoader",
    "PluginManager",
    "PluginMetadata",
    "PluginNotFoundError",
    "PluginRegistry",
    "PluginSource",
    "PluginState",
    "version_matches",
]
