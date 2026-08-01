"""Canonical base model for persistent production assets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from .reference import ReferenceImage


class AssetType(StrEnum):
    CHARACTER = "character"
    ENVIRONMENT = "environment"
    WARDROBE = "wardrobe"
    PROP = "prop"
    VEHICLE = "vehicle"
    STORYBOARD = "storyboard"
    REFERENCE = "reference"


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class AssetVersion:
    version: int
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    reference_images: list[ReferenceImage] = field(default_factory=list)

    def validate(self) -> list[str]:
        errors = [] if self.version > 0 else ["version number must be positive"]
        for reference in self.reference_images:
            errors.extend(reference.validate())
        return errors


@dataclass(slots=True, kw_only=True)
class Asset:
    """Stable, versioned identity shared by every canonical asset."""

    name: str
    asset_id: UUID = field(default_factory=uuid4)
    description: str = ""
    version: int = 1
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str | None = None
    content_hash: str = ""
    reference_images: list[ReferenceImage] = field(default_factory=list)
    versions: list[AssetVersion] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.asset_id, str):
            self.asset_id = UUID(self.asset_id)
        if self.updated_at is None:
            self.updated_at = self.created_at
        if not self.content_hash:
            self.refresh_content_hash()

    @property
    def kind(self) -> str:
        return getattr(type(self), "asset_type", type(self).__name__.lower())

    @property
    def references(self) -> list[ReferenceImage]:
        return self.reference_images

    def add_reference(
        self, file_path: str, *, label: str = "", **values: Any
    ) -> ReferenceImage:
        if label and "notes" not in values:
            values["notes"] = label
        reference = ReferenceImage(file_path=file_path, **values)
        self.reference_images.append(reference)
        self.touch()
        return reference

    def create_version(self, note: str = "") -> AssetVersion:
        revision = AssetVersion(
            len(self.versions) + 1,
            note,
            dict(self.metadata),
            [item.copy() for item in self.reference_images],
        )
        self.versions.append(revision)
        self.version = revision.version
        self.touch()
        return revision

    def touch(self) -> None:
        self.updated_at = _now()
        self.refresh_content_hash()

    def hash_payload(self) -> dict[str, Any]:
        return {
            "asset_id": str(self.asset_id),
            "type": self.kind,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "tags": sorted(self.tags),
            "metadata": self.metadata,
            "references": [item.to_dict() for item in self.reference_images],
        }

    def refresh_content_hash(self) -> str:
        self.content_hash = self.calculate_content_hash()
        return self.content_hash

    def calculate_content_hash(self) -> str:
        """Return the canonical hash without changing the asset.

        Keeping calculation separate from ``refresh_content_hash`` lets validators
        detect manifests whose identity metadata was edited without updating the
        recorded digest.
        """
        data = json.dumps(
            self.hash_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.asset_id, UUID):
            errors.append("asset_id must be a UUID")
        if not self.name.strip():
            errors.append("asset name cannot be empty")
        if self.version < 1:
            errors.append("asset version must be positive")
        if self.content_hash != self.calculate_content_hash():
            errors.append("asset content hash does not match canonical content")
        if any(not tag.strip() for tag in self.tags):
            errors.append("asset tags cannot be empty")
        numbers = [item.version for item in self.versions]
        if numbers != list(range(1, len(numbers) + 1)):
            errors.append("asset versions must be consecutive starting at 1")
        for reference in self.reference_images:
            errors.extend(reference.validate())
        return errors
