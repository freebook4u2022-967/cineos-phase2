"""Plugin registration, discovery, and lifecycle orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import metadata as importlib_metadata

from .errors import PluginLifecycleError, PluginLoadError, PluginRegistrationError
from .plugin import Plugin, PluginContext, PluginMetadata

PLUGIN_ENTRY_POINT_GROUP = "cineos.plugins"
SUPPORTED_API_VERSION = "1"


class PluginManager:
    """Own a deterministic collection of plugin instances."""

    def __init__(self, *, api_version: str = SUPPORTED_API_VERSION) -> None:
        self.api_version = api_version
        self._plugins: dict[str, Plugin] = {}
        self._active: set[str] = set()
        self._disabled: set[str] = set()
        self._activating: set[str] = set()

    @property
    def plugins(self) -> tuple[Plugin, ...]:
        """Registered plugins, ordered by their stable names."""

        return tuple(self._plugins[name] for name in sorted(self._plugins))

    @property
    def active_plugins(self) -> tuple[Plugin, ...]:
        """Active plugins, ordered by their stable names."""

        return tuple(self._plugins[name] for name in sorted(self._active))

    @property
    def enabled_plugins(self) -> tuple[Plugin, ...]:
        """Enabled plugins, ordered by their stable names."""

        return tuple(
            plugin
            for plugin in self.plugins
            if plugin.metadata.name not in self._disabled
        )

    def enable(self, name: str) -> Plugin:
        """Enable a registered plugin without activating it."""

        plugin = self.get(name)
        self._disabled.discard(name)
        return plugin

    def disable(self, name: str, context: PluginContext | None = None) -> Plugin:
        """Disable a plugin, deactivating it and active dependants first."""

        plugin = self.get(name)
        resolved_context = context or PluginContext()
        dependant_names = {name}
        changed = True
        while changed:
            changed = False
            for candidate in self.plugins:
                candidate_name = candidate.metadata.name
                if (
                    candidate_name in self._active
                    and candidate_name not in dependant_names
                    and dependant_names.intersection(candidate.metadata.dependencies)
                ):
                    dependant_names.add(candidate_name)
                    changed = True
        dependants = sorted(dependant_names - {name}, reverse=True)
        for dependant in dependants:
            self.deactivate(dependant, resolved_context)
        self.deactivate(name, resolved_context)
        self._disabled.add(name)
        return plugin

    def register(self, plugin: Plugin | type[Plugin]) -> Plugin:
        """Register an instance (or a zero-argument plugin class)."""

        try:
            instance = plugin() if isinstance(plugin, type) else plugin
        except Exception as error:
            raise PluginRegistrationError("could not instantiate plugin") from error

        metadata = getattr(instance, "metadata", None)
        if not isinstance(instance, Plugin) or not isinstance(metadata, PluginMetadata):
            raise PluginRegistrationError(
                "plugin must inherit Plugin and declare PluginMetadata"
            )
        if metadata.api_version != self.api_version:
            raise PluginRegistrationError(
                f"plugin {metadata.name!r} requires API {metadata.api_version!r}; "
                f"host provides {self.api_version!r}"
            )
        if metadata.name in self._plugins:
            raise PluginRegistrationError(
                f"plugin {metadata.name!r} is already registered"
            )
        self._plugins[metadata.name] = instance
        return instance

    def unregister(self, name: str, context: PluginContext | None = None) -> Plugin:
        """Unregister a plugin, deactivating it first when necessary."""

        plugin = self.get(name)
        if name in self._active:
            self.deactivate(name, context or PluginContext())
        del self._plugins[name]
        self._disabled.discard(name)
        return plugin

    def get(self, name: str) -> Plugin:
        """Return a registered plugin by name."""

        try:
            return self._plugins[name]
        except KeyError as error:
            raise PluginRegistrationError(
                f"plugin {name!r} is not registered"
            ) from error

    def discover(
        self,
        entry_points: Iterable[importlib_metadata.EntryPoint] | None = None,
    ) -> tuple[Plugin, ...]:
        """Load plugins from the ``cineos.plugins`` entry-point group.

        Entry points can be injected to make discovery deterministic in tests and
        embedded hosts. Discovery order is normalized by entry-point name.
        """

        if entry_points is None:
            entry_points = importlib_metadata.entry_points(
                group=PLUGIN_ENTRY_POINT_GROUP
            )
        loaded: list[Plugin] = []
        for entry_point in sorted(entry_points, key=lambda item: item.name):
            try:
                candidate = entry_point.load()
                loaded.append(self.register(candidate))
            except PluginRegistrationError:
                raise
            except Exception as error:
                raise PluginLoadError(
                    f"could not load plugin entry point {entry_point.name!r}"
                ) from error
        return tuple(loaded)

    def activate(self, name: str, context: PluginContext | None = None) -> Plugin:
        """Activate a registered plugin exactly once."""

        plugin = self.get(name)
        if name in self._active:
            return plugin
        if name in self._disabled:
            raise PluginLifecycleError(f"plugin {name!r} is disabled")
        if name in self._activating:
            raise PluginLifecycleError(f"plugin dependency cycle includes {name!r}")
        resolved_context = context or PluginContext()
        self._activating.add(name)
        try:
            for dependency in plugin.metadata.dependencies:
                if dependency not in self._plugins:
                    raise PluginLifecycleError(
                        f"plugin {name!r} requires unregistered plugin {dependency!r}"
                    )
                self.activate(dependency, resolved_context)
            plugin.activate(resolved_context)
        except Exception as error:
            if isinstance(error, PluginLifecycleError):
                raise
            raise PluginLifecycleError(f"plugin {name!r} failed to activate") from error
        finally:
            self._activating.remove(name)
        self._active.add(name)
        return plugin

    def deactivate(self, name: str, context: PluginContext | None = None) -> Plugin:
        """Deactivate an active plugin exactly once."""

        plugin = self.get(name)
        if name not in self._active:
            return plugin
        try:
            plugin.deactivate(context or PluginContext())
        except Exception as error:
            raise PluginLifecycleError(
                f"plugin {name!r} failed to deactivate"
            ) from error
        self._active.remove(name)
        return plugin

    def activate_all(self, context: PluginContext | None = None) -> None:
        """Activate every plugin in deterministic order, rolling back on failure."""

        resolved_context = context or PluginContext()
        previously_active = set(self._active)
        try:
            for plugin in self.enabled_plugins:
                if plugin.metadata.name not in self._active:
                    self.activate(plugin.metadata.name, resolved_context)
        except PluginLifecycleError:
            for name in sorted(self._active - previously_active, reverse=True):
                self.deactivate(name, resolved_context)
            raise

    def deactivate_all(self, context: PluginContext | None = None) -> None:
        """Deactivate every active plugin in reverse deterministic order."""

        resolved_context = context or PluginContext()
        for name in sorted(self._active, reverse=True):
            self.deactivate(name, resolved_context)
