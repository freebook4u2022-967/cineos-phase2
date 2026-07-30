"""Plugin registration, dependency resolution, and lifecycle management."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .discovery import (
    candidates_from_modules,
    discover_directory,
    discover_entry_points,
)
from .exceptions import (
    PluginCompatibilityError,
    PluginDependencyError,
    PluginLifecycleError,
    PluginValidationError,
)
from .metadata import PluginMetadata, parse_version, version_satisfies
from .plugin import Plugin

PLUGIN_API_VERSION = "1.0.0"


@dataclass(slots=True)
class _PluginState:
    plugin: Plugin
    enabled: bool = False


class PluginManager:
    """Own plugin instances without coupling them to a renderer or application."""

    def __init__(self, *, context: Any = None, api_version: str = PLUGIN_API_VERSION):
        parse_version(api_version)
        self.context = context
        self.api_version = api_version
        self._plugins: dict[str, _PluginState] = {}

    @property
    def plugins(self) -> tuple[Plugin, ...]:
        """Return loaded plugins in deterministic insertion order."""
        return tuple(state.plugin for state in self._plugins.values())

    def metadata(self, name: str) -> PluginMetadata:
        return self._state(name).plugin.metadata

    def is_enabled(self, name: str) -> bool:
        return self._state(name).enabled

    def get(self, name: str) -> Plugin:
        return self._state(name).plugin

    def load(self, candidate: Any, *, enable: bool = True) -> Plugin:
        """Instantiate, validate, register, and optionally enable one plugin."""
        plugin = self._instantiate(candidate)
        metadata = getattr(plugin, "metadata", None)
        if not isinstance(metadata, PluginMetadata):
            raise PluginValidationError("plugin.metadata must be PluginMetadata")
        if metadata.name in self._plugins:
            raise PluginValidationError(f"plugin {metadata.name!r} is already loaded")
        if parse_version(metadata.api_version)[0] != parse_version(self.api_version)[0]:
            raise PluginCompatibilityError(
                f"plugin {metadata.name!r} requires API {metadata.api_version}; "
                f"framework provides {self.api_version}"
            )
        self._check_dependencies(metadata)
        self._plugins[metadata.name] = _PluginState(plugin)
        try:
            plugin.on_load(self.context)
            if enable:
                self.enable(metadata.name)
        except Exception as error:
            self._plugins.pop(metadata.name, None)
            if isinstance(error, (PluginDependencyError, PluginLifecycleError)):
                raise
            raise PluginLifecycleError(
                f"plugin {metadata.name!r} failed to load"
            ) from error
        return plugin

    def load_many(
        self, candidates: Iterable[Any], *, enable: bool = True
    ) -> tuple[Plugin, ...]:
        """Load candidates in dependency order, rejecting unresolved graphs."""
        pending = list(candidates)
        loaded: list[Plugin] = []
        while pending:
            progress = False
            failures: list[PluginDependencyError] = []
            for candidate in pending.copy():
                try:
                    plugin = self.load(candidate, enable=enable)
                except PluginDependencyError as error:
                    failures.append(error)
                    continue
                pending.remove(candidate)
                loaded.append(plugin)
                progress = True
            if not progress:
                detail = "; ".join(str(error) for error in failures)
                raise PluginDependencyError(f"unresolved plugin dependencies: {detail}")
        return tuple(loaded)

    def discover_and_load(
        self, path: str | None = None, *, enable: bool = True
    ) -> tuple[Plugin, ...]:
        """Discover installed entry points or modules in ``path`` and load them."""
        if path is None:
            candidates = discover_entry_points().values()
        else:
            candidates = candidates_from_modules(discover_directory(path).values())
        return self.load_many(candidates, enable=enable)

    def enable(self, name: str) -> None:
        state = self._state(name)
        if state.enabled:
            return
        self._check_dependencies(state.plugin.metadata)
        try:
            state.plugin.on_enable()
        except Exception as error:
            raise PluginLifecycleError(f"plugin {name!r} failed to enable") from error
        state.enabled = True

    def disable(self, name: str) -> None:
        state = self._state(name)
        if not state.enabled:
            return
        dependents = self._enabled_dependents(name)
        if dependents:
            raise PluginDependencyError(
                f"cannot disable {name!r}; enabled dependents: {', '.join(dependents)}"
            )
        try:
            state.plugin.on_disable()
        except Exception as error:
            raise PluginLifecycleError(f"plugin {name!r} failed to disable") from error
        state.enabled = False

    def unload(self, name: str) -> Plugin:
        """Disable and remove a plugin after ensuring nothing depends on it."""
        state = self._state(name)
        dependents = [
            plugin_name
            for plugin_name, other in self._plugins.items()
            if name in other.plugin.metadata.dependencies
        ]
        if dependents:
            raise PluginDependencyError(
                f"cannot unload {name!r}; loaded dependents: {', '.join(dependents)}"
            )
        if state.enabled:
            self.disable(name)
        try:
            state.plugin.on_unload()
        except Exception as error:
            raise PluginLifecycleError(f"plugin {name!r} failed to unload") from error
        del self._plugins[name]
        return state.plugin

    def unload_all(self) -> None:
        """Unload all plugins in reverse dependency/load order."""
        while self._plugins:
            available = [
                name
                for name in self._plugins
                if not any(
                    name in state.plugin.metadata.dependencies
                    for state in self._plugins.values()
                )
            ]
            if not available:
                raise PluginDependencyError(
                    "cannot unload cyclic plugin dependency graph"
                )
            for name in reversed(available):
                self.unload(name)

    def _check_dependencies(self, metadata: PluginMetadata) -> None:
        for name, constraint in metadata.dependencies.items():
            dependency = self._plugins.get(name)
            if dependency is None:
                raise PluginDependencyError(
                    f"plugin {metadata.name!r} requires missing plugin {name!r}"
                )
            if not dependency.enabled:
                raise PluginDependencyError(
                    f"plugin {metadata.name!r} requires enabled plugin {name!r}"
                )
            if not version_satisfies(dependency.plugin.metadata.version, constraint):
                raise PluginDependencyError(
                    f"plugin {metadata.name!r} requires {name!r} {constraint}; "
                    f"found {dependency.plugin.metadata.version}"
                )

    def _enabled_dependents(self, name: str) -> list[str]:
        return [
            plugin_name
            for plugin_name, state in self._plugins.items()
            if state.enabled and name in state.plugin.metadata.dependencies
        ]

    def _state(self, name: str) -> _PluginState:
        try:
            return self._plugins[name]
        except KeyError as error:
            raise PluginValidationError(f"plugin {name!r} is not loaded") from error

    @staticmethod
    def _instantiate(candidate: Any) -> Plugin:
        if isinstance(candidate, Plugin):
            return candidate
        if callable(candidate):
            plugin = candidate()
            if isinstance(plugin, Plugin):
                return plugin
        raise PluginValidationError(
            "plugin candidate must be a Plugin or zero-argument factory"
        )
