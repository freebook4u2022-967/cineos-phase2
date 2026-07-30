"""Exceptions raised by the CINEOS plugin framework."""


class PluginError(Exception):
    """Base class for plugin framework failures."""


class PluginDiscoveryError(PluginError):
    """A plugin candidate could not be discovered or imported."""


class PluginValidationError(PluginError):
    """A plugin does not satisfy the plugin contract."""


class PluginCompatibilityError(PluginValidationError):
    """A plugin is incompatible with this framework version."""


class PluginDependencyError(PluginValidationError):
    """A plugin dependency is missing, disabled, or incompatible."""


class PluginLifecycleError(PluginError):
    """A plugin lifecycle callback failed."""
