"""Exceptions raised by the CINEOS plugin framework."""


class PluginError(Exception):
    """Base class for plugin framework failures."""


class PluginValidationError(PluginError, ValueError):
    """A plugin or its metadata does not satisfy the plugin contract."""


class PluginCompatibilityError(PluginValidationError):
    """A plugin targets an incompatible framework API version."""


class PluginDependencyError(PluginError):
    """A plugin dependency is missing or prevents an operation."""


class PluginNotFoundError(PluginError, LookupError):
    """A requested plugin is not registered."""
