"""Persist and load canonical JSON Film Packages."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .manifest import FilmPackage
from .serializer import deserialize, package_from_dict, serialize
from .validator import verify


def save(package: FilmPackage, destination: str | Path | None = None) -> str:
    """Verify and serialize a package, optionally writing it to *destination*."""

    verify(package)
    data = serialize(package)
    if destination is not None:
        Path(destination).write_text(data + "\n", encoding="utf-8")
    return data


def load(source: str | bytes | Path | Mapping[str, Any]) -> FilmPackage:
    """Load and verify a package from JSON, a path, or a decoded mapping."""

    if isinstance(source, Mapping):
        package = package_from_dict(source)
    elif isinstance(source, Path):
        package = deserialize(source.read_bytes())
    elif isinstance(source, str):
        candidate = Path(source)
        if not source.lstrip().startswith(("{", "[")) and candidate.is_file():
            package = deserialize(candidate.read_bytes())
        else:
            package = deserialize(source)
    elif isinstance(source, bytes):
        package = deserialize(source)
    else:
        raise TypeError("source must be JSON, a path, or a mapping")
    verify(package)
    return package
