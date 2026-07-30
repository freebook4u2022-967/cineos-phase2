"""Atlas runtime configuration loading."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    workers: int = 1
    log_level: str = "INFO"
    shutdown_timeout: float = 5.0

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.shutdown_timeout < 0:
            raise ValueError("shutdown_timeout cannot be negative")


def load_config(
    path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> RuntimeConfig:
    """Load JSON configuration, with ``CINEOS_RUNTIME_*`` environment overrides."""
    values: dict[str, Any] = {}
    if path is not None:
        with Path(path).open(encoding="utf-8") as config_file:
            values.update(json.load(config_file))
    env = os.environ if environ is None else environ
    mappings: dict[str, tuple[str, type[str] | type[int] | type[float]]] = {
        "CINEOS_RUNTIME_WORKERS": ("workers", int),
        "CINEOS_RUNTIME_LOG_LEVEL": ("log_level", str),
        "CINEOS_RUNTIME_SHUTDOWN_TIMEOUT": ("shutdown_timeout", float),
    }
    for environment_name, (field_name, converter) in mappings.items():
        if environment_name in env:
            values[field_name] = converter(env[environment_name])
    known = {"workers", "log_level", "shutdown_timeout"}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown runtime configuration: {', '.join(sorted(unknown))}")
    return RuntimeConfig(**values)
