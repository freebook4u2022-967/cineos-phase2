"""Canonical asset collection and relationship resolution."""

from __future__ import annotations

from uuid import UUID

from .base import Asset
from .exceptions import AssetNotFoundError, DuplicateAssetError
from .relationship import AssetRelationship

_RULES = {
    "character-wardrobe": ("character", "wardrobe"),
    "character-prop": ("character", "prop"),
    "character-vehicle": ("character", "vehicle"),
    "storyboard-scene": ("storyboard", "scene"),
    "scene-environment": ("scene", "environment"),
    # Historical spellings remain loadable.
    "wears": ("character", "wardrobe"),
    "uses": ("character", "prop"),
    "drives": ("character", "vehicle"),
}


class AssetRegistry:
    def __init__(self) -> None:
        self._assets: dict[UUID, Asset] = {}
        self._relationships: set[AssetRelationship] = set()

    def validate_uniqueness(
        self, asset: Asset, *, replacing: UUID | None = None
    ) -> None:
        if asset.asset_id in self._assets and asset.asset_id != replacing:
            raise DuplicateAssetError(f"duplicate asset UUID: {asset.asset_id}")
        for existing in self._assets.values():
            if existing.asset_id != replacing and (
                existing.kind.casefold(),
                existing.name.casefold(),
            ) == (asset.kind.casefold(), asset.name.casefold()):
                raise DuplicateAssetError(
                    f"duplicate {asset.kind} asset name: {asset.name!r}"
                )

    def register(self, asset: Asset) -> Asset:
        self.validate_uniqueness(asset)
        self._assets[asset.asset_id] = asset
        return asset

    def update(self, asset: Asset) -> Asset:
        if asset.asset_id not in self._assets:
            raise AssetNotFoundError(str(asset.asset_id))
        self.validate_uniqueness(asset, replacing=asset.asset_id)
        asset.touch()
        self._assets[asset.asset_id] = asset
        return asset

    def retrieve(self, asset_id: UUID | str) -> Asset:
        try:
            return self._assets[UUID(str(asset_id))]
        except (KeyError, ValueError) as error:
            raise AssetNotFoundError(str(asset_id)) from error

    get = retrieve

    def list(self, *, kind: str | None = None, tag: str | None = None) -> list[Asset]:
        values = self._assets.values()
        if kind:
            values = (item for item in values if item.kind == kind.lower())
        if tag:
            values = (item for item in values if tag in item.tags)
        return sorted(
            values, key=lambda item: (item.kind, item.name, str(item.asset_id))
        )

    def search_by_tag(self, tag: str) -> list[Asset]:
        return self.list(tag=tag)

    def remove(self, asset_id: UUID | str) -> Asset:
        key = UUID(str(asset_id))
        asset = self.retrieve(key)
        del self._assets[key]
        self._relationships = {
            item
            for item in self._relationships
            if item.source_id != key and item.target_id != key
        }
        return asset

    def relate(
        self, source: Asset | UUID | str, target: Asset | UUID | str, relationship: str
    ) -> AssetRelationship:
        source_id = source.asset_id if isinstance(source, Asset) else UUID(str(source))
        target_id = target.asset_id if isinstance(target, Asset) else UUID(str(target))
        if source_id not in self._assets or target_id not in self._assets:
            raise ValueError("both relationship endpoints must be registered")
        if not relationship.strip():
            raise ValueError("relationship type cannot be empty")
        value = AssetRelationship(source_id, target_id, relationship.strip())
        self._relationships.add(value)
        return value

    @property
    def relationships(self) -> tuple[AssetRelationship, ...]:
        return tuple(
            sorted(
                self._relationships,
                key=lambda x: (str(x.source_id), x.relationship, str(x.target_id)),
            )
        )

    def resolve_relationships(
        self, asset: Asset | UUID | str, relationship: str | None = None
    ) -> list[Asset]:
        key = asset.asset_id if isinstance(asset, Asset) else UUID(str(asset))
        ids = {
            item.target_id
            for item in self._relationships
            if item.source_id == key
            and (relationship is None or item.relationship == relationship)
        }
        return sorted((self._assets[item] for item in ids), key=lambda item: item.name)

    related_to = resolve_relationships

    def validate(self) -> list[str]:
        errors: list[str] = []
        names: set[tuple[str, str]] = set()
        for asset in self.list():
            errors.extend(f"asset {asset.asset_id}: {e}" for e in asset.validate())
            key = (asset.kind.casefold(), asset.name.casefold())
            if key in names:
                errors.append(f"duplicate {asset.kind} asset name: {asset.name!r}")
            names.add(key)
        for item in self.relationships:
            source = self._assets.get(item.source_id)
            target = self._assets.get(item.target_id)
            if source is None or target is None:
                errors.append(
                    f"relationship {item.relationship!r} has unknown endpoint"
                )
            elif (
                item.relationship in _RULES
                and (source.kind, target.kind) != _RULES[item.relationship]
            ):
                errors.append(
                    f"invalid {item.relationship} relationship: "
                    f"{source.kind} to {target.kind}"
                )
        return errors

    def __len__(self) -> int:
        return len(self._assets)
