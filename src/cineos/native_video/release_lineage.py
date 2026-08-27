"""Fail-closed lineage for production releases and safe model/runtime upgrades.

A CINEOS production release is already cryptographically bound to its final-film
QC evidence. Long-lived production also needs a durable upgrade chain so an
operator can prove which audited release superseded which prior release without
silently skipping or forking history.

This module deliberately stores only immutable fingerprints and an optional
migration fingerprint. It does not execute migrations; it provides the trust
boundary that migration tooling must satisfy before activation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .audited_release import AuditedProductionRelease
from .release_receipt import ProductionReleaseError, canonical_sha256

PRODUCTION_RELEASE_LINEAGE_SCHEMA = "cineos-production-release-lineage/0.1"
GENESIS_PREVIOUS_RELEASE = "0" * 64


def _require_sha256(field_name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value.lower())
    ):
        raise ProductionReleaseError(f"{field_name} must be one SHA-256 hex digest")
    return value.lower()


@dataclass(frozen=True, slots=True)
class ProductionReleaseLineageEntry:
    """One immutable node in the production-release upgrade chain."""

    sequence: int
    release_fingerprint: str
    previous_release_fingerprint: str
    model_fingerprint: str
    runtime_fingerprint: str
    migration_fingerprint: str
    schema: str = PRODUCTION_RELEASE_LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PRODUCTION_RELEASE_LINEAGE_SCHEMA:
            raise ProductionReleaseError("unsupported production release lineage schema")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 0:
            raise ProductionReleaseError("sequence must be a non-negative integer")
        for field_name in (
            "release_fingerprint",
            "previous_release_fingerprint",
            "model_fingerprint",
            "runtime_fingerprint",
            "migration_fingerprint",
        ):
            object.__setattr__(self, field_name, _require_sha256(field_name, getattr(self, field_name)))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.as_dict())


def create_release_lineage_entry(
    release: AuditedProductionRelease,
    *,
    previous: ProductionReleaseLineageEntry | None = None,
    migration_fingerprint: str | None = None,
) -> ProductionReleaseLineageEntry:
    """Create the next lineage node for an audited release.

    Genesis releases use a zero previous-release digest. Every later release must
    point to the exact audited release fingerprint represented by the prior node.
    A migration fingerprint is mandatory whenever model or runtime identity changes.
    """
    if not isinstance(release, AuditedProductionRelease):
        raise TypeError("release must be AuditedProductionRelease")
    if previous is not None and not isinstance(previous, ProductionReleaseLineageEntry):
        raise TypeError("previous must be ProductionReleaseLineageEntry or None")

    if previous is None:
        sequence = 0
        previous_release_fingerprint = GENESIS_PREVIOUS_RELEASE
        changed_composition = False
    else:
        sequence = previous.sequence + 1
        previous_release_fingerprint = previous.release_fingerprint
        changed_composition = (
            previous.model_fingerprint != release.model_fingerprint
            or previous.runtime_fingerprint != release.runtime_fingerprint
        )

    if migration_fingerprint is None:
        migration_digest = GENESIS_PREVIOUS_RELEASE
    else:
        migration_digest = _require_sha256("migration_fingerprint", migration_fingerprint)

    if changed_composition and migration_digest == GENESIS_PREVIOUS_RELEASE:
        raise ProductionReleaseError(
            "model/runtime upgrade requires a migration_fingerprint"
        )
    if not changed_composition and migration_digest != GENESIS_PREVIOUS_RELEASE:
        raise ProductionReleaseError(
            "migration_fingerprint supplied without model/runtime composition change"
        )

    return ProductionReleaseLineageEntry(
        sequence=sequence,
        release_fingerprint=release.fingerprint,
        previous_release_fingerprint=previous_release_fingerprint,
        model_fingerprint=release.model_fingerprint,
        runtime_fingerprint=release.runtime_fingerprint,
        migration_fingerprint=migration_digest,
    )


def verify_release_lineage(
    entries: Iterable[ProductionReleaseLineageEntry],
) -> tuple[ProductionReleaseLineageEntry, ...]:
    """Verify one complete, gap-free, fork-free release lineage."""
    chain = tuple(entries)
    if not chain:
        raise ProductionReleaseError("production release lineage cannot be empty")

    seen_release_fingerprints: set[str] = set()
    for index, entry in enumerate(chain):
        if not isinstance(entry, ProductionReleaseLineageEntry):
            raise TypeError("lineage entries must be ProductionReleaseLineageEntry")
        if entry.sequence != index:
            raise ProductionReleaseError("production release lineage sequence gap")
        if entry.release_fingerprint in seen_release_fingerprints:
            raise ProductionReleaseError("production release lineage contains a duplicate release")
        seen_release_fingerprints.add(entry.release_fingerprint)

        if index == 0:
            if entry.previous_release_fingerprint != GENESIS_PREVIOUS_RELEASE:
                raise ProductionReleaseError("genesis release must use zero previous-release digest")
            if entry.migration_fingerprint != GENESIS_PREVIOUS_RELEASE:
                raise ProductionReleaseError("genesis release cannot declare a migration")
            continue

        previous = chain[index - 1]
        if entry.previous_release_fingerprint != previous.release_fingerprint:
            raise ProductionReleaseError("production release lineage previous-release mismatch")
        changed_composition = (
            previous.model_fingerprint != entry.model_fingerprint
            or previous.runtime_fingerprint != entry.runtime_fingerprint
        )
        has_migration = entry.migration_fingerprint != GENESIS_PREVIOUS_RELEASE
        if changed_composition and not has_migration:
            raise ProductionReleaseError("release upgrade missing migration fingerprint")
        if not changed_composition and has_migration:
            raise ProductionReleaseError("spurious migration fingerprint in release lineage")

    return chain


__all__ = [
    "GENESIS_PREVIOUS_RELEASE",
    "PRODUCTION_RELEASE_LINEAGE_SCHEMA",
    "ProductionReleaseLineageEntry",
    "create_release_lineage_entry",
    "verify_release_lineage",
]
