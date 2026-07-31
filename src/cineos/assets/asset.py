"""Common values for versioned production assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4


@dataclass(slots=True)
class ReferenceImage:
    """A local or remote image used as visual guidance for an asset."""

    uri: str
    label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> list[str]:
        return [] if self.uri.strip() else ["reference image URI cannot be empty"]


@dataclass(slots=True)
class AssetVersion:
    """An immutable-in-practice snapshot label and its revision metadata."""

    version: int
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    reference_images: list[ReferenceImage] = field(default_factory=list)

    def validate(self) -> list[str]:
        errors = [] if self.version > 0 else ["version number must be positive"]
        for image in self.reference_images:
            errors.extend(image.validate())
        return errors


@dataclass(slots=True, kw_only=True)
class Asset:
    """Base class for an identified, tagged, and versioned production asset."""

    name: str
    asset_id: UUID = field(default_factory=uuid4)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: set[str] = field(default_factory=set)
    reference_images: list[ReferenceImage] = field(default_factory=list)
    versions: list[AssetVersion] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return type(self).__name__.lower()

    def add_reference(self, uri: str, *, label: str = "", **metadata: Any) -> None:
        self.reference_images.append(ReferenceImage(uri, label, metadata))

    def create_version(self, note: str = "") -> AssetVersion:
        """Capture the current metadata and references as the next revision."""

        revision = AssetVersion(
            version=len(self.versions) + 1,
            note=note,
            metadata=dict(self.metadata),
            reference_images=[
                ReferenceImage(image.uri, image.label, dict(image.metadata))
                for image in self.reference_images
            ],
        )
        self.versions.append(revision)
        return revision

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not isinstance(self.asset_id, UUID):
            errors.append("asset_id must be a UUID")
        if not self.name.strip():
            errors.append("asset name cannot be empty")
        if any(not tag.strip() for tag in self.tags):
            errors.append("asset tags cannot be empty")
        numbers = [revision.version for revision in self.versions]
        if numbers != list(range(1, len(numbers) + 1)):
            errors.append("asset versions must be consecutive starting at 1")
        for image in self.reference_images:
            errors.extend(image.validate())
        for revision in self.versions:
            errors.extend(revision.validate())
        return errors
