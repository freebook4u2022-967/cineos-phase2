"""Versioned, renderer-independent Film Package value type."""

from dataclasses import dataclass, field
from typing import Any

FILM_PACKAGE_VERSION = "1.0"


@dataclass(slots=True)
class FilmPackage:
    """Deterministic compilation output for a :class:`MovieProject`."""

    project_metadata: dict[str, Any]
    scene_manifest: list[dict[str, Any]]
    shot_manifest: list[dict[str, Any]]
    character_manifest: list[dict[str, Any]]
    location_manifest: list[dict[str, Any]]
    asset_manifest: list[dict[str, Any]]
    timeline_manifest: dict[str, Any]
    content_hashes: dict[str, str] = field(default_factory=dict)
    version: str = FILM_PACKAGE_VERSION
