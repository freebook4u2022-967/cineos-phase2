"""JSON serialization for Film Packages."""

import json
from collections.abc import Mapping
from typing import Any

from .hashing import canonical_json
from .manifest import FilmPackage


def package_to_dict(package: FilmPackage) -> dict[str, Any]:
    """Convert a package into plain JSON-compatible values."""

    return {
        "version": package.version,
        "project_metadata": package.project_metadata,
        "scene_manifest": package.scene_manifest,
        "shot_manifest": package.shot_manifest,
        "character_manifest": package.character_manifest,
        "location_manifest": package.location_manifest,
        "asset_manifest": package.asset_manifest,
        "timeline_manifest": package.timeline_manifest,
        "content_hashes": package.content_hashes,
        "cinedna_ids": package.cinedna_ids,
    }


def package_from_dict(value: Mapping[str, Any]) -> FilmPackage:
    """Construct a Film Package from a decoded JSON object."""

    return FilmPackage(
        version=value.get("version", ""),
        project_metadata=dict(value.get("project_metadata", {})),
        scene_manifest=list(value.get("scene_manifest", [])),
        shot_manifest=list(value.get("shot_manifest", [])),
        character_manifest=list(value.get("character_manifest", [])),
        location_manifest=list(value.get("location_manifest", [])),
        asset_manifest=list(value.get("asset_manifest", [])),
        timeline_manifest=dict(value.get("timeline_manifest", {})),
        content_hashes=dict(value.get("content_hashes", {})),
        cinedna_ids=list(value.get("cinedna_ids", [])),
    )


def serialize(package: FilmPackage) -> str:
    """Serialize a package to canonical JSON."""

    return canonical_json(package_to_dict(package))


def deserialize(data: str | bytes | bytearray) -> FilmPackage:
    """Deserialize UTF-8 JSON into a Film Package."""

    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Film Package JSON must contain an object")
    return package_from_dict(value)
