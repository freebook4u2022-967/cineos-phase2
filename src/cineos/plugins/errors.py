"""Errors raised by the CINEOS plugin framework."""


class PluginError(Exception):
    """Base class for plugin framework failures."""


class PluginRegistrationError(PluginError):
    """Raised when a plugin cannot be registered."""


class PluginLoadError(PluginError):
    """Raised when a discovered plugin cannot be loaded."""


class PluginLifecycleError(PluginError):
    """Raised when a plugin lifecycle callback fails."""
