"""In-memory registry for loaded plugins."""

from collections.abc import Iterator

from .base import Plugin
from .exceptions import DuplicatePluginError, PluginNotFoundError


class PluginRegistry:
    """Store loaded plugin instances by their stable metadata name."""

    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        name = plugin.metadata.name
        if name in self._plugins:
            raise DuplicatePluginError(f"Plugin {name!r} is already registered")
        self._plugins[name] = plugin

    def unregister(self, name: str) -> Plugin:
        try:
            return self._plugins.pop(name)
        except KeyError as error:
            raise PluginNotFoundError(name) from error

    def get(self, name: str) -> Plugin:
        try:
            return self._plugins[name]
        except KeyError as error:
            raise PluginNotFoundError(name) from error

    def contains(self, name: str) -> bool:
        return name in self._plugins

    def __contains__(self, name: object) -> bool:
        return name in self._plugins

    def __iter__(self) -> Iterator[Plugin]:
        return iter(tuple(self._plugins.values()))

    def __len__(self) -> int:
        return len(self._plugins)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._plugins)
