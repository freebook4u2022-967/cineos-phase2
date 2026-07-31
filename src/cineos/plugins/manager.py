"""Plugin discovery and lifecycle management."""

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Any

from .errors import (
    PluginCompatibilityError,
    PluginDependencyError,
    PluginNotFoundError,
    PluginValidationError,
)
from .metadata import PluginMetadata, version_tuple
from .plugin import Plugin

PLUGIN_API_VERSION = "1.0.0"
ENTRY_POINT_GROUP = "cineos.plugins"


@dataclass(slots=True)
class _PluginState:
    plugin: Plugin
    loaded: bool = False
    enabled: bool = False


class PluginManager:
    """Discover and operate plugins while enforcing lifecycle invariants."""

    def __init__(self, *, context: Any = None, api_version: str = PLUGIN_API_VERSION):
        version_tuple(api_version)
        self.context = context
        self.api_version = api_version
        self._plugins: dict[str, _PluginState] = {}
        self._loading: set[str] = set()

    @property
    def plugins(self) -> tuple[Plugin, ...]:
        """Registered plugins in deterministic name order."""
        return tuple(self._plugins[name].plugin for name in sorted(self._plugins))

    @property
    def loaded_plugins(self) -> tuple[Plugin, ...]:
        return tuple(
            plugin for plugin in self.plugins if self.is_loaded(plugin.metadata.name)
        )

    @property
    def enabled_plugins(self) -> tuple[Plugin, ...]:
        return tuple(
            plugin for plugin in self.plugins if self.is_enabled(plugin.metadata.name)
        )

    def register(self, plugin: Plugin) -> Plugin:
        metadata = getattr(plugin, "metadata", None)
        if not isinstance(metadata, PluginMetadata):
            raise PluginValidationError(
                "plugin metadata must be a PluginMetadata instance"
            )
        if version_tuple(metadata.api_version)[0] != version_tuple(self.api_version)[0]:
            raise PluginCompatibilityError(
                f"plugin {metadata.name!r} targets API {metadata.api_version}; "
                f"host provides {self.api_version}"
            )
        if metadata.name in self._plugins:
            raise PluginValidationError(
                f"plugin {metadata.name!r} is already registered"
            )
        self._plugins[metadata.name] = _PluginState(plugin)
        return plugin

    def discover(self, *, group: str = ENTRY_POINT_GROUP) -> tuple[Plugin, ...]:
        """Discover, instantiate, and register Python entry-point plugins."""
        discovered: list[Plugin] = []
        points = importlib_metadata.entry_points()
        selected = (
            points.select(group=group)
            if hasattr(points, "select")
            else points.get(group, ())
        )
        for point in sorted(selected, key=lambda item: item.name):
            candidate = point.load()
            plugin = candidate() if isinstance(candidate, type) else candidate
            discovered.append(self.register(plugin))
        return tuple(discovered)

    def register_all(self, plugins: Iterable[Plugin]) -> tuple[Plugin, ...]:
        return tuple(self.register(plugin) for plugin in plugins)

    def get(self, name: str) -> Plugin:
        return self._state(name).plugin

    def is_loaded(self, name: str) -> bool:
        return self._state(name).loaded

    def is_enabled(self, name: str) -> bool:
        return self._state(name).enabled

    def load(self, name: str) -> Plugin:
        state = self._state(name)
        if state.loaded:
            return state.plugin
        if name in self._loading:
            raise PluginDependencyError(
                f"circular plugin dependency involving {name!r}"
            )
        self._loading.add(name)
        try:
            for dependency in state.plugin.metadata.dependencies:
                if dependency not in self._plugins:
                    raise PluginDependencyError(
                        f"plugin {name!r} requires unregistered plugin {dependency!r}"
                    )
                self.load(dependency)
            state.plugin.on_load(self.context)
            state.loaded = True
        finally:
            self._loading.remove(name)
        return state.plugin

    def enable(self, name: str) -> Plugin:
        state = self._state(name)
        self.load(name)
        if state.enabled:
            return state.plugin
        for dependency in state.plugin.metadata.dependencies:
            self.enable(dependency)
        state.plugin.on_enable(self.context)
        state.enabled = True
        return state.plugin

    def disable(self, name: str) -> Plugin:
        state = self._state(name)
        if not state.enabled:
            return state.plugin
        dependents = self._active_dependents(name)
        if dependents:
            raise PluginDependencyError(
                f"cannot disable {name!r}; enabled dependents: {', '.join(dependents)}"
            )
        state.plugin.on_disable(self.context)
        state.enabled = False
        return state.plugin

    def unload(self, name: str) -> Plugin:
        state = self._state(name)
        dependents = self._loaded_dependents(name)
        if dependents:
            raise PluginDependencyError(
                f"cannot unload {name!r}; loaded dependents: {', '.join(dependents)}"
            )
        if state.enabled:
            self.disable(name)
        if state.loaded:
            state.plugin.on_unload(self.context)
            state.loaded = False
        return state.plugin

    def unregister(self, name: str) -> Plugin:
        plugin = self.unload(name)
        del self._plugins[name]
        return plugin

    def _state(self, name: str) -> _PluginState:
        try:
            return self._plugins[name]
        except KeyError as error:
            raise PluginNotFoundError(f"plugin {name!r} is not registered") from error

    def _active_dependents(self, name: str) -> list[str]:
        return sorted(
            key
            for key, state in self._plugins.items()
            if state.enabled and name in state.plugin.metadata.dependencies
        )

    def _loaded_dependents(self, name: str) -> list[str]:
        return sorted(
            key
            for key, state in self._plugins.items()
            if state.loaded and name in state.plugin.metadata.dependencies
        )
