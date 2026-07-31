"""Deterministic JSON encoding for asset registries."""

from __future__ import annotations

import json
from typing import Any


def dumps(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(value, sort_keys=True, indent=indent, ensure_ascii=False) + "\n"


def loads(data: str | bytes | bytearray) -> Any:
    return json.loads(data)
