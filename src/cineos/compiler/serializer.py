"""JSON serialization for Film Packages."""

from __future__ import annotations

import json
from typing import Any

from .hashing import canonical_json
from .manifest import FilmPackage


def dumps(package: FilmPackage, *, pretty: bool = False) -> str:
    """Serialize a Film Package; compact output is canonical and deterministic."""
    if not pretty:
        return canonical_json(package.to_dict())
    return (
        json.dumps(
            package.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def loads(data: str | bytes | bytearray) -> FilmPackage:
    """Deserialize JSON into a Film Package without implicitly trusting it."""
    try:
        value: Any = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid Film Package JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("a Film Package must be a JSON object")
    return FilmPackage.from_dict(value)
