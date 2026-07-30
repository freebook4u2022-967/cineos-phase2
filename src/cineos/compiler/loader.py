"""Filesystem persistence for Film Packages."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .manifest import FilmPackage
from .serializer import dumps, loads
from .validator import validate


def save(package: FilmPackage, destination: str | Path) -> Path:
    """Validate and atomically save *package* as canonical JSON."""
    validate(package)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(dumps(package))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def load(source: str | Path) -> FilmPackage:
    """Load and verify a Film Package JSON file."""
    package = loads(Path(source).read_bytes())
    validate(package)
    return package
