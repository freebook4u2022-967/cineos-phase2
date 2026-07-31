"""Plugin identity and compatibility metadata."""

import re
from dataclasses import dataclass

from .errors import PluginValidationError

_VERSION = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+].+)?$")


def version_tuple(version: str) -> tuple[int, int, int]:
    """Return the numeric semantic-version core."""
    match = _VERSION.fullmatch(version)
    if match is None:
        raise PluginValidationError(f"invalid semantic version: {version!r}")
    return tuple(int(part) for part in match.groups())


@dataclass(frozen=True, slots=True)
class PluginMetadata:
    """Immutable metadata supplied by every plugin.

    Dependencies are plugin names. They intentionally do not name renderers or
    other product subsystems, keeping the contract usable by every CINEOS host.
    """

    name: str
    version: str
    api_version: str
    description: str = ""
    author: str = ""
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise PluginValidationError("plugin name must not be empty")
        if self.name != self.name.strip():
            raise PluginValidationError(
                "plugin name must not have surrounding whitespace"
            )
        version_tuple(self.version)
        version_tuple(self.api_version)
        dependencies = tuple(self.dependencies)
        if any(not item or item != item.strip() for item in dependencies):
            raise PluginValidationError("dependency names must not be empty")
        if self.name in dependencies:
            raise PluginValidationError("a plugin cannot depend on itself")
        if len(set(dependencies)) != len(dependencies):
            raise PluginValidationError("plugin dependencies must be unique")
        object.__setattr__(self, "dependencies", dependencies)
