"""Deterministic MovieProject to FilmPackage compiler."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from .hashing import canonical_json, canonicalize, content_hash
from .manifest import FILM_PACKAGE_VERSION, FilmPackage
from .validator import package_hash, validate

_FIELDS = {
    "project_metadata": ("project_metadata", "metadata"),
    "scene_manifest": ("scene_manifest", "scenes"),
    "shot_manifest": ("shot_manifest", "shots"),
    "character_manifest": ("character_manifest", "characters"),
    "location_manifest": ("location_manifest", "locations"),
    "asset_manifest": ("asset_manifest", "assets"),
    "timeline_manifest": ("timeline_manifest", "timeline"),
}


def _read(project: Any, aliases: tuple[str, ...], default: Any) -> Any:
    for name in aliases:
        if isinstance(project, Mapping) and name in project:
            return project[name]
        if hasattr(project, name):
            return getattr(project, name)
    return default


def _ordered_manifest(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be a sequence")
    normalized = canonicalize(value)
    return sorted(normalized, key=canonical_json)


def compile(project: Any) -> FilmPackage:
    """Compile a mapping, dataclass, or MovieProject-like object deterministically."""
    if is_dataclass(project) and not isinstance(project, type):
        project = asdict(project)
    if not isinstance(project, Mapping) and not hasattr(project, "__dict__"):
        raise TypeError(
            "project must be a mapping, dataclass, or MovieProject-like object"
        )

    values: dict[str, Any] = {}
    metadata = canonicalize(_read(project, _FIELDS["project_metadata"], {}))
    if not isinstance(metadata, dict):
        raise ValueError("project metadata must be an object")
    values["project_metadata"] = metadata
    for name in (
        "scene_manifest",
        "shot_manifest",
        "character_manifest",
        "location_manifest",
        "asset_manifest",
    ):
        values[name] = _ordered_manifest(_read(project, _FIELDS[name], []), name)
    values["timeline_manifest"] = canonicalize(
        _read(project, _FIELDS["timeline_manifest"], [])
    )

    hashes = {name: content_hash(value) for name, value in values.items()}
    package = FilmPackage(**values, content_hashes=hashes, version=FILM_PACKAGE_VERSION)
    hashes["package"] = package_hash(package)
    validate(package)
    return package
