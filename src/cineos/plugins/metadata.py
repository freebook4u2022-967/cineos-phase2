"""Plugin metadata and lightweight semantic-version matching."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$")
_CONSTRAINT_RE = re.compile(r"^(==|!=|>=|<=|>|<|~=)?\s*(\d+\.\d+\.\d+)$")


def _version_tuple(version: str) -> tuple[int, int, int]:
    match = _VERSION_RE.fullmatch(version.strip())
    if not match:
        raise ValueError(f"Invalid semantic version: {version!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def version_matches(version: str, specifier: str) -> bool:
    """Return whether a semantic version satisfies comma-separated constraints.

    Supported operators are ``==``, ``!=``, ``>=``, ``<=``, ``>``, ``<`` and
    compatible release ``~=``. An empty specifier accepts every valid version.
    """

    actual = _version_tuple(version)
    for raw_constraint in filter(None, (item.strip() for item in specifier.split(","))):
        match = _CONSTRAINT_RE.fullmatch(raw_constraint)
        if not match:
            raise ValueError(f"Invalid version constraint: {raw_constraint!r}")
        operator = match.group(1) or "=="
        expected = _version_tuple(match.group(2))
        comparisons = {
            "==": actual == expected,
            "!=": actual != expected,
            ">=": actual >= expected,
            "<=": actual <= expected,
            ">": actual > expected,
            "<": actual < expected,
            "~=": actual >= expected and actual < (expected[0], expected[1] + 1, 0),
        }
        if not comparisons[operator]:
            return False
    return True


@dataclass(frozen=True, slots=True)
class PluginMetadata:
    """Declarative identity, compatibility, and dependency information."""

    name: str
    version: str
    description: str = ""
    author: str = ""
    cineos_version: str = ""
    dependencies: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Plugin name must not be empty")
        _version_tuple(self.version)
        if self.cineos_version:
            version_matches("0.0.0", self.cineos_version)
        for name, constraint in self.dependencies.items():
            if not name or not name.strip():
                raise ValueError("Dependency names must not be empty")
            version_matches("0.0.0", constraint)
        object.__setattr__(
            self, "dependencies", MappingProxyType(dict(self.dependencies))
        )

    def supports_cineos(self, version: str) -> bool:
        """Return whether the metadata declares support for *version*."""

        return version_matches(version, self.cineos_version)
