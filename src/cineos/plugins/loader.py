"""Discovery and construction of plugins from Python modules and entry points."""

import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Iterable
from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
from types import ModuleType
from typing import Any

from .base import Plugin
from .exceptions import PluginDiscoveryError, PluginLoadError
from .metadata import PluginMetadata

PluginSource = Plugin | type[Plugin] | EntryPoint | str | Path


class PluginLoader:
    """Find plugin providers and turn providers into plugin instances."""

    def __init__(self, entry_point_group: str = "cineos.plugins") -> None:
        self.entry_point_group = entry_point_group

    def discover(self, paths: Iterable[str | Path] = ()) -> tuple[PluginSource, ...]:
        """Discover installed entry points and importable plugins in directories."""

        discovered: list[PluginSource] = list(
            entry_points().select(group=self.entry_point_group)
        )
        for location in paths:
            path = Path(location)
            if not path.is_dir():
                raise PluginDiscoveryError(f"Plugin path is not a directory: {path}")
            discovered.extend(
                child
                for child in sorted(path.iterdir())
                if child.suffix == ".py" and child.name != "__init__.py"
            )
            discovered.extend(
                child
                for child in sorted(path.iterdir())
                if child.is_dir() and (child / "__init__.py").is_file()
            )
        return tuple(discovered)

    def load(self, source: PluginSource) -> Plugin:
        """Construct a plugin from an instance, class, module, file, or entry point."""

        try:
            provider: Any = source
            if isinstance(source, EntryPoint):
                provider = source.load()
            elif isinstance(source, Path):
                provider = self._load_path(source)
            elif isinstance(source, str):
                candidate = Path(source)
                provider = (
                    self._load_path(candidate)
                    if candidate.exists()
                    else importlib.import_module(source)
                )
            if isinstance(provider, ModuleType):
                provider = self._provider_from_module(provider)
            if isinstance(provider, type) and issubclass(provider, Plugin):
                provider = provider()
            if not isinstance(provider, Plugin):
                raise TypeError("provider is not a Plugin instance or Plugin subclass")
            if not isinstance(getattr(provider, "metadata", None), PluginMetadata):
                raise TypeError("plugin metadata is not a PluginMetadata instance")
            return provider
        except PluginLoadError:
            raise
        except Exception as error:
            raise PluginLoadError(
                f"Could not load plugin from {source!r}: {error}"
            ) from error

    def _load_path(self, path: Path) -> ModuleType:
        path = path.resolve()
        module_file = path / "__init__.py" if path.is_dir() else path
        if not module_file.is_file():
            raise PluginDiscoveryError(f"Plugin module does not exist: {path}")
        digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
        module_name = f"_cineos_plugin_{digest}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            module_file,
            submodule_search_locations=[str(path)] if path.is_dir() else None,
        )
        if spec is None or spec.loader is None:
            raise PluginDiscoveryError(f"Cannot import plugin module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    @staticmethod
    def _provider_from_module(module: ModuleType) -> Any:
        for export in ("plugin", "PLUGIN"):
            if hasattr(module, export):
                return getattr(module, export)
        classes = [
            value
            for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, Plugin)
            and value is not Plugin
            and value.__module__ == module.__name__
        ]
        if len(classes) != 1:
            raise PluginDiscoveryError(
                f"Module {module.__name__!r} must export 'plugin'/'PLUGIN' or "
                "define exactly one Plugin subclass"
            )
        return classes[0]
