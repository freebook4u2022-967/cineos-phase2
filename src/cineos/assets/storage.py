"""Filesystem persistence for metadata and media paths (never media bytes)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from .base import Asset, AssetVersion
from .character import Character
from .environment import Environment
from .prop import Prop
from .reference import ReferenceImage
from .reference_asset import GenericReference
from .registry import AssetRegistry
from .serializer import dumps, loads
from .storyboard import Storyboard
from .vehicle import Vehicle
from .wardrobe import Wardrobe

ASSET_FORMAT = "cineos-assets-v1"
_TYPES: dict[str, type[Asset]] = {
    cls.asset_type: cls
    for cls in (
        Character,
        Environment,
        Wardrobe,
        Prop,
        Vehicle,
        Storyboard,
        GenericReference,
    )
}


def _reference(value: dict[str, Any]) -> ReferenceImage:
    # Accept the small Phase 1 reference shape as a migration convenience.
    if "uri" in value:
        return ReferenceImage(
            uri=value["uri"],
            label=value.get("label", ""),
            metadata=value.get("metadata"),
        )
    return ReferenceImage(**value)


def asset_to_dict(asset: Asset) -> dict[str, Any]:
    asset.refresh_content_hash()
    return {
        "asset_id": str(asset.asset_id),
        "type": asset.kind,
        "name": asset.name,
        "description": asset.description,
        "version": asset.version,
        "tags": sorted(asset.tags),
        "metadata": asset.metadata,
        "created_at": asset.created_at,
        "updated_at": asset.updated_at,
        "content_hash": asset.content_hash,
        "references": [item.to_dict() for item in asset.references],
        "versions": [
            {
                "version": revision.version,
                "note": revision.note,
                "metadata": revision.metadata,
                "references": [item.to_dict() for item in revision.reference_images],
            }
            for revision in asset.versions
        ],
    }


def registry_to_dict(registry: AssetRegistry) -> dict[str, Any]:
    return {
        "format": ASSET_FORMAT,
        "assets": [asset_to_dict(asset) for asset in registry.list()],
        "relationships": [
            {
                "source_id": str(item.source_id),
                "target_id": str(item.target_id),
                "relationship": item.relationship,
            }
            for item in registry.relationships
        ],
    }


def registry_from_dict(value: dict[str, Any]) -> AssetRegistry:
    if value.get("format", ASSET_FORMAT) != ASSET_FORMAT:
        raise ValueError("unsupported asset registry format")
    registry = AssetRegistry()
    for item in value.get("assets", []):
        kind = item.get("type")
        if kind not in _TYPES:
            raise ValueError(f"unsupported asset type: {kind!r}")
        references = [
            _reference(image)
            for image in item.get("references", item.get("reference_images", []))
        ]
        versions = [
            AssetVersion(
                version=revision["version"],
                note=revision.get("note", ""),
                metadata=dict(revision.get("metadata", {})),
                reference_images=[
                    _reference(image)
                    for image in revision.get(
                        "references", revision.get("reference_images", [])
                    )
                ],
            )
            for revision in item.get("versions", [])
        ]
        registry.register(
            _TYPES[kind](
                asset_id=UUID(item["asset_id"]),
                name=item["name"],
                description=item.get("description", ""),
                version=item.get("version", max(1, len(versions))),
                tags=set(item.get("tags", [])),
                metadata=dict(item.get("metadata", {})),
                created_at=item.get("created_at") or "1970-01-01T00:00:00Z",
                updated_at=item.get("updated_at"),
                content_hash=item.get("content_hash", ""),
                reference_images=references,
                versions=versions,
            )
        )
    for item in value.get("relationships", []):
        registry.relate(item["source_id"], item["target_id"], item["relationship"])
    return registry


def save(registry: AssetRegistry, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(dumps(registry_to_dict(registry)), encoding="utf-8")
    return destination


def load(path: str | Path) -> AssetRegistry:
    value = loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("asset registry JSON must contain an object")
    return registry_from_dict(value)


class FileSystemAssetStorage:
    """Small repository façade whose JSON contains paths, never copied media."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, registry: AssetRegistry) -> Path:
        return save(registry, self.path)

    def load(self) -> AssetRegistry:
        return load(self.path)
