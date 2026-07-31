"""Renderer-independent plugin contracts."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class PluginMetadata:
    """Stable identity and compatibility information for a plugin."""

    name: str
    version: str
    description: str = ""
    api_version: str = "1"
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "api_version"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"plugin {field_name} must not be empty")
        if any(not dependency.strip() for dependency in self.dependencies):
            raise ValueError("plugin dependency names must not be empty")
        if self.name in self.dependencies:
            raise ValueError("plugin must not depend on itself")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("plugin dependencies must be unique")


@dataclass(frozen=True, slots=True)
class PluginContext:
    """Host-owned values supplied to lifecycle callbacks.

    Values are intentionally generic: plugins integrate with host services through
    explicit objects rather than acquiring a renderer or another subsystem.
    """

    services: Mapping[str, Any] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "services", MappingProxyType(dict(self.services)))
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))


class Plugin(ABC):
    """Base contract for optional CINEOS extensions.

    Subclasses declare metadata and may override either lifecycle method. The base
    implementation has no renderer, compiler, runtime, or CLI dependency.
    """

    metadata: PluginMetadata

    def activate(self, context: PluginContext) -> None:
        """Activate the plugin for a host context."""

    def deactivate(self, context: PluginContext) -> None:
        """Release resources acquired during activation."""
