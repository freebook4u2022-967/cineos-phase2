"""Film Package data model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FILM_PACKAGE_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class FilmPackage:
    """A deterministic, renderer-independent compiled movie project."""

    project_metadata: dict[str, Any]
    scene_manifest: list[Any]
    shot_manifest: list[Any]
    character_manifest: list[Any]
    location_manifest: list[Any]
    asset_manifest: list[Any]
    timeline_manifest: Any
    content_hashes: dict[str, str]
    version: str = FILM_PACKAGE_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the wire-format representation of this package."""
        return {
            "version": self.version,
            "project_metadata": self.project_metadata,
            "scene_manifest": self.scene_manifest,
            "shot_manifest": self.shot_manifest,
            "character_manifest": self.character_manifest,
            "location_manifest": self.location_manifest,
            "asset_manifest": self.asset_manifest,
            "timeline_manifest": self.timeline_manifest,
            "content_hashes": self.content_hashes,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FilmPackage:
        """Build a package from its wire-format representation."""
        return cls(
            version=value.get("version", ""),
            project_metadata=value.get("project_metadata"),
            scene_manifest=value.get("scene_manifest"),
            shot_manifest=value.get("shot_manifest"),
            character_manifest=value.get("character_manifest"),
            location_manifest=value.get("location_manifest"),
            asset_manifest=value.get("asset_manifest"),
            timeline_manifest=value.get("timeline_manifest"),
            content_hashes=value.get("content_hashes"),
        )
