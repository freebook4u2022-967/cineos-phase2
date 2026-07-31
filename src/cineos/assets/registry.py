"""Asset collection, relationship, and validation services."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from .asset import Asset


@dataclass(frozen=True, slots=True)
class AssetRelationship:
    """A directed, typed association between two assets."""

    source_id: UUID
    target_id: UUID
    relationship: str


class AssetRegistry:
    """Own assets and their cross-asset relationships."""

    def __init__(self) -> None:
        self._assets: dict[UUID, Asset] = {}
        self._relationships: set[AssetRelationship] = set()

    def register(self, asset: Asset) -> Asset:
        if asset.asset_id in self._assets:
            raise ValueError(f"duplicate asset UUID: {asset.asset_id}")
        self._assets[asset.asset_id] = asset
        return asset

    def get(self, asset_id: UUID | str) -> Asset:
        return self._assets[UUID(str(asset_id))]

    def list(self, *, kind: str | None = None, tag: str | None = None) -> list[Asset]:
        assets = self._assets.values()
        if kind is not None:
            assets = (asset for asset in assets if asset.kind == kind.lower())
        if tag is not None:
            assets = (asset for asset in assets if tag in asset.tags)
        return sorted(
            assets, key=lambda asset: (asset.kind, asset.name, str(asset.asset_id))
        )

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
                key=lambda item: (
                    str(item.source_id),
                    item.relationship,
                    str(item.target_id),
                ),
            )
        )

    def related_to(
        self, asset: Asset | UUID | str, relationship: str | None = None
    ) -> list[Asset]:
        asset_id = asset.asset_id if isinstance(asset, Asset) else UUID(str(asset))
        ids = {
            item.target_id
            for item in self._relationships
            if item.source_id == asset_id
            and (relationship is None or item.relationship == relationship)
        }
        return sorted((self._assets[item] for item in ids), key=lambda item: item.name)

    def validate(self) -> list[str]:
        errors: list[str] = []
        for asset in self.list():
            errors.extend(
                f"asset {asset.asset_id}: {error}" for error in asset.validate()
            )
        for item in self.relationships:
            if item.source_id not in self._assets or item.target_id not in self._assets:
                errors.append(
                    f"relationship {item.relationship!r} has unknown endpoint"
                )
        return errors

    def __len__(self) -> int:
        return len(self._assets)
