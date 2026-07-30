"""Plugin discovery through Python entry points and explicit directories."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import metadata as importlib_metadata
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

from .exceptions import PluginDiscoveryError

ENTRY_POINT_GROUP = "cineos.plugins"


def discover_entry_points(group: str = ENTRY_POINT_GROUP) -> dict[str, Any]:
    """Load installed plugin candidates advertised in ``group``."""
    discovered: dict[str, Any] = {}
    entry_points = importlib_metadata.entry_points()
    selected = entry_points.select(group=group)
    for entry_point in sorted(selected, key=lambda item: item.name):
        try:
            discovered[entry_point.name] = entry_point.load()
        except Exception as error:
            raise PluginDiscoveryError(
                f"could not load plugin entry point {entry_point.name!r}"
            ) from error
    return discovered


def discover_directory(path: str | Path) -> dict[str, ModuleType]:
    """Import non-private Python modules from an explicitly trusted directory."""
    directory = Path(path)
    if not directory.is_dir():
        raise PluginDiscoveryError(f"plugin directory does not exist: {directory}")
    discovered: dict[str, ModuleType] = {}
    for source in sorted(directory.glob("*.py")):
        if source.name.startswith("_"):
            continue
        module_name = f"cineos_discovered_plugin_{source.stem}"
        spec = spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise PluginDiscoveryError(f"could not create module spec for {source}")
        module = module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            raise PluginDiscoveryError(
                f"could not import plugin module {source}"
            ) from error
        discovered[source.stem] = module
    return discovered


def candidates_from_modules(modules: Iterable[ModuleType]) -> list[Any]:
    """Extract the conventional ``plugin`` object or ``create_plugin`` factory."""
    candidates = []
    for module in modules:
        if hasattr(module, "create_plugin"):
            candidates.append(module.create_plugin)
        elif hasattr(module, "plugin"):
            candidates.append(module.plugin)
        else:
            raise PluginDiscoveryError(
                f"module {module.__name__!r} exposes neither plugin nor create_plugin"
            )
    return candidates
