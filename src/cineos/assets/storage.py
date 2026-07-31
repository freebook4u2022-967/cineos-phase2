"""Portable JSON storage for asset registries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from .asset import Asset, AssetVersion, ReferenceImage
from .character import Character
from .environment import Environment
from .prop import Prop
from .registry import AssetRegistry
from .storyboard import Storyboard
from .vehicle import Vehicle
from .wardrobe import Wardrobe

ASSET_FORMAT = "cineos-assets-v1"
_TYPES: dict[str, type[Asset]] = {
    value.__name__.lower(): value
    for value in (Character, Environment, Prop, Vehicle, Wardrobe, Storyboard)
}


def _reference_to_dict(image: ReferenceImage) -> dict[str, Any]:
    return {"uri": image.uri, "label": image.label, "metadata": image.metadata}


def asset_to_dict(asset: Asset) -> dict[str, Any]:
    """Convert an asset to JSON-compatible values."""

    return {
        "asset_id": str(asset.asset_id),
        "type": asset.kind,
        "name": asset.name,
        "description": asset.description,
        "metadata": asset.metadata,
        "tags": sorted(asset.tags),
        "reference_images": [
            _reference_to_dict(item) for item in asset.reference_images
        ],
        "versions": [
            {
                "version": item.version,
                "note": item.note,
                "metadata": item.metadata,
                "reference_images": [
                    _reference_to_dict(image) for image in item.reference_images
                ],
            }
            for item in asset.versions
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
            ReferenceImage(**image) for image in item.get("reference_images", [])
        ]
        versions = [
            AssetVersion(
                version=revision["version"],
                note=revision.get("note", ""),
                metadata=dict(revision.get("metadata", {})),
                reference_images=[
                    ReferenceImage(**image)
                    for image in revision.get("reference_images", [])
                ],
            )
            for revision in item.get("versions", [])
        ]
        registry.register(
            _TYPES[kind](
                asset_id=UUID(item["asset_id"]),
                name=item["name"],
                description=item.get("description", ""),
                metadata=dict(item.get("metadata", {})),
                tags=set(item.get("tags", [])),
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
    destination.write_text(
        json.dumps(registry_to_dict(registry), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def load(path: str | Path) -> AssetRegistry:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("asset registry JSON must contain an object")
    return registry_from_dict(value)
