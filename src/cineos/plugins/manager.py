"""Plugin lifecycle orchestration."""

from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

from cineos import __version__

from .base import Plugin
from .exceptions import (
    PluginCompatibilityError,
    PluginDependencyError,
    PluginLifecycleError,
)
from .loader import PluginLoader, PluginSource
from .metadata import version_matches
from .registry import PluginRegistry


class PluginState(StrEnum):
    """Manager-owned lifecycle state for a loaded plugin."""

    INITIALIZED = "initialized"
    ENABLED = "enabled"


class PluginManager:
    """Load plugins and enforce compatibility, dependencies, and lifecycle order."""

    def __init__(
        self,
        *,
        cineos_version: str = __version__,
        loader: PluginLoader | None = None,
        registry: PluginRegistry | None = None,
    ) -> None:
        self.cineos_version = cineos_version
        self.loader = loader if loader is not None else PluginLoader()
        self.registry = registry if registry is not None else PluginRegistry()
        self._states: dict[str, PluginState] = {}

    def discover(self, paths: Iterable[str | Path] = ()) -> tuple[PluginSource, ...]:
        """Return providers found by the configured loader."""

        return self.loader.discover(paths)

    def load(self, source: PluginSource) -> Plugin:
        """Load, validate, register, and initialize one plugin."""

        plugin = self.loader.load(source)
        metadata = plugin.metadata
        if not metadata.supports_cineos(self.cineos_version):
            raise PluginCompatibilityError(
                f"Plugin {metadata.name!r} requires CINEOS "
                f"{metadata.cineos_version!r}; running {self.cineos_version}"
            )
        self._validate_dependencies(plugin)
        self.registry.register(plugin)
        try:
            plugin.initialize()
        except Exception as error:
            self.registry.unregister(metadata.name)
            raise PluginLifecycleError(
                f"Plugin {metadata.name!r} failed to initialize"
            ) from error
        self._states[metadata.name] = PluginState.INITIALIZED
        return plugin

    def enable(self, name: str) -> None:
        """Start an initialized plugin after ensuring dependencies are enabled."""

        plugin = self.registry.get(name)
        if self._states[name] is PluginState.ENABLED:
            return
        disabled_dependencies = [
            dependency
            for dependency in plugin.metadata.dependencies
            if self._states.get(dependency) is not PluginState.ENABLED
        ]
        if disabled_dependencies:
            joined = ", ".join(disabled_dependencies)
            raise PluginDependencyError(
                f"Plugin {name!r} requires enabled dependencies: {joined}"
            )
        try:
            plugin.start()
        except Exception as error:
            raise PluginLifecycleError(f"Plugin {name!r} failed to start") from error
        self._states[name] = PluginState.ENABLED

    def disable(self, name: str) -> None:
        """Stop an enabled plugin if no enabled plugin depends on it."""

        plugin = self.registry.get(name)
        if self._states[name] is PluginState.INITIALIZED:
            return
        dependents = self._enabled_dependents(name)
        if dependents:
            raise PluginDependencyError(
                f"Cannot disable {name!r}; enabled plugins depend on it: "
                + ", ".join(dependents)
            )
        try:
            plugin.stop()
        except Exception as error:
            raise PluginLifecycleError(f"Plugin {name!r} failed to stop") from error
        self._states[name] = PluginState.INITIALIZED

    def unload(self, name: str) -> Plugin:
        """Stop, shut down, and unregister a plugin."""

        dependents = [
            plugin.metadata.name
            for plugin in self.registry
            if name in plugin.metadata.dependencies
        ]
        if dependents:
            raise PluginDependencyError(
                f"Cannot unload {name!r}; loaded plugins depend on it: "
                + ", ".join(dependents)
            )
        plugin = self.registry.get(name)
        if self._states[name] is PluginState.ENABLED:
            self.disable(name)
        try:
            plugin.shutdown()
        except Exception as error:
            raise PluginLifecycleError(
                f"Plugin {name!r} failed to shut down"
            ) from error
        self._states.pop(name)
        return self.registry.unregister(name)

    def is_enabled(self, name: str) -> bool:
        """Return whether a loaded plugin is active."""

        self.registry.get(name)
        return self._states[name] is PluginState.ENABLED

    def _validate_dependencies(self, plugin: Plugin) -> None:
        for name, constraint in plugin.metadata.dependencies.items():
            if name not in self.registry:
                raise PluginDependencyError(
                    f"Plugin {plugin.metadata.name!r} requires missing plugin {name!r}"
                )
            installed = self.registry.get(name).metadata.version
            if not version_matches(installed, constraint):
                raise PluginDependencyError(
                    f"Plugin {plugin.metadata.name!r} requires "
                    f"{name!r} {constraint!r}; "
                    f"loaded version is {installed}"
                )

    def _enabled_dependents(self, name: str) -> list[str]:
        return [
            plugin.metadata.name
            for plugin in self.registry
            if name in plugin.metadata.dependencies
            and self._states[plugin.metadata.name] is PluginState.ENABLED
        ]
