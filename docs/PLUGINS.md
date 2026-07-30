# CINEOS Plugin Framework

The plugin framework provides discovery, loading, dependency validation, and a
predictable lifecycle for optional CINEOS extensions. It is infrastructure
only: plugins do not receive renderer, AI, or GPU capabilities from this API.

## Defining a plugin

Subclass `cineos.plugins.Plugin` and supply immutable `PluginMetadata`:

```python
from cineos.plugins import Plugin, PluginMetadata


class EditorialPlugin(Plugin):
    metadata = PluginMetadata(
        name="editorial",
        version="1.2.0",
        cineos_version=">=0.1.0,<1.0.0",
        dependencies={"asset-catalog": ">=2.0.0,<3.0.0"},
    )

    def initialize(self) -> None:
        # Allocate inactive resources.
        ...

    def start(self) -> None:
        # Begin providing the plugin service.
        ...

    def stop(self) -> None:
        # Stop providing the service, retaining resources.
        ...

    def shutdown(self) -> None:
        # Release all resources.
        ...
```

Versions use three-part semantic versions. Compatibility and dependency
constraints may combine `==`, `!=`, `>=`, `<=`, `>`, `<`, and `~=` with
commas. For example, `>=1.1.0,<2.0.0` accepts the 1.x releases from 1.1.0.

## Publishing and discovery

Installed distributions can advertise a plugin class or instance through the
`cineos.plugins` entry-point group:

```toml
[project.entry-points."cineos.plugins"]
editorial = "cineos_editorial:EditorialPlugin"
```

`PluginLoader.discover()` returns those entry points. Directories passed to
`discover(paths)` are also scanned for Python files and packages. A file or
package must export `plugin` or `PLUGIN`, or define exactly one local `Plugin`
subclass. Discovery does not initialize or enable code; loading a discovered
provider performs the import and construction.

## Managing lifecycle

```python
from cineos.plugins import PluginManager

manager = PluginManager()
providers = manager.discover(["./project_plugins"])
plugin = manager.load(providers[0])   # validates, registers, initializes
manager.enable(plugin.metadata.name) # starts the plugin
manager.disable(plugin.metadata.name) # stops the plugin
manager.unload(plugin.metadata.name)  # shuts down and unregisters
```

Load dependencies before their dependents, and enable dependencies before
enabling their dependents. The manager rejects missing or incompatible
dependencies. It also prevents disabling or unloading a plugin while a loaded
dependent would be invalidated. Lifecycle calls are idempotent where no state
transition is needed.

If a lifecycle hook raises, the manager wraps the failure in
`PluginLifecycleError`. An initialization failure rolls registry changes back.
Other failures preserve the prior managed state so callers can inspect or retry
the operation.

## API responsibilities

- `Plugin` defines lifecycle hooks.
- `PluginMetadata` defines identity, host compatibility, and dependencies.
- `PluginLoader` discovers providers and constructs plugin instances.
- `PluginRegistry` stores unique loaded instances by name.
- `PluginManager` validates and orchestrates state transitions.
- `cineos.plugins.exceptions` exposes errors suitable for targeted handling.
