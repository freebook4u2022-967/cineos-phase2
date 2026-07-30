"""Exceptions raised by the CINEOS plugin framework."""


class PluginError(Exception):
    """Base class for all plugin framework errors."""


class PluginDiscoveryError(PluginError):
    """Raised when a plugin cannot be discovered or imported."""


class PluginLoadError(PluginError):
    """Raised when a plugin cannot be loaded or unloaded safely."""


class PluginNotFoundError(PluginError, KeyError):
    """Raised when a requested plugin is not registered."""


class DuplicatePluginError(PluginError):
    """Raised when a plugin name is already registered."""


class PluginCompatibilityError(PluginLoadError):
    """Raised when a plugin does not support this CINEOS version."""


class PluginDependencyError(PluginLoadError):
    """Raised when plugin dependencies are absent or incompatible."""


class PluginLifecycleError(PluginError):
    """Raised when a lifecycle operation fails or is invalid."""
