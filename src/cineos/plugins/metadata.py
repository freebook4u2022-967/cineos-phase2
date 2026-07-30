"""Declarative metadata and version constraints for plugins."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .exceptions import PluginValidationError

_VERSION = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$"
)
_OPERATORS = (">=", "<=", "!=", "==", ">", "<")


def parse_version(value: str) -> tuple[int, int, int]:
    """Return the numeric components of a strict semantic version."""
    match = _VERSION.fullmatch(value)
    if match is None:
        raise PluginValidationError(f"invalid semantic version: {value!r}")
    return tuple(int(match.group(part)) for part in ("major", "minor", "patch"))


def version_satisfies(version: str, constraint: str) -> bool:
    """Evaluate a comma-separated set of simple semantic-version comparisons."""
    candidate = parse_version(version)
    for raw_clause in constraint.split(","):
        clause = raw_clause.strip()
        operator = next((item for item in _OPERATORS if clause.startswith(item)), "==")
        requested = (
            clause[len(operator) :].strip() if clause.startswith(operator) else clause
        )
        target = parse_version(requested)
        comparisons = {
            "==": candidate == target,
            "!=": candidate != target,
            ">=": candidate >= target,
            "<=": candidate <= target,
            ">": candidate > target,
            "<": candidate < target,
        }
        if not comparisons[operator]:
            return False
    return True


@dataclass(frozen=True, slots=True)
class PluginMetadata:
    """Stable, renderer-independent identity and requirements for a plugin."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    api_version: str = "1.0.0"
    dependencies: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise PluginValidationError("plugin name must not be empty")
        if self.name != self.name.strip():
            raise PluginValidationError(
                "plugin name must not have surrounding whitespace"
            )
        parse_version(self.version)
        parse_version(self.api_version)
        dependencies = dict(self.dependencies)
        if self.name in dependencies:
            raise PluginValidationError("a plugin cannot depend on itself")
        for name, constraint in dependencies.items():
            if not name or not constraint:
                raise PluginValidationError(
                    "dependency names and constraints must not be empty"
                )
            version_satisfies("0.0.0", constraint)
        object.__setattr__(self, "dependencies", MappingProxyType(dependencies))
