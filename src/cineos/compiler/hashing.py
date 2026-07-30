"""Canonical JSON encoding and hashing utilities for Film Packages."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class CanonicalizationError(ValueError):
    """Raised when a value cannot be represented by the package format."""


def canonicalize(value: Any) -> Any:
    """Convert supported Python values to deterministic JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return canonicalize(asdict(value))
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("object keys must be strings")
            result[key] = canonicalize(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [canonicalize(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("non-finite floats are not valid JSON")
        return value
    if isinstance(value, Path):
        return str(value)
    raise CanonicalizationError(f"unsupported value type: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize *value* using the canonical Film Package JSON representation."""
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_hash(value: Any) -> str:
    """Return a lowercase SHA-256 digest for a canonicalized value."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
