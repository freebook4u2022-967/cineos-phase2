"""Explicit migration path for durable CINEOS production runtime manifests.

Production checkpoints may outlive the software version that created them. The
strict :class:`ProductionRuntimeManifest` restore path intentionally rejects unknown
schemas and fields; weakening that validator to accommodate upgrades would make old
runtimes silently ignore new production semantics. This module keeps the strict
validator intact and provides a separate, auditable migration boundary.

Migrations are deliberately one-way and deterministic. Each source schema may have
at most one registered successor, every migration must declare the exact schema it
produces, cycles are rejected at runtime, and the final payload is always parsed by
``ProductionRuntimeManifest.restore``. A missing migration fails closed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from .runtime_manifest import (
    PRODUCTION_RUNTIME_MANIFEST_SCHEMA,
    ProductionRuntimeManifest,
)

RuntimeManifestMigration = Callable[[dict[str, object]], dict[str, object]]


class RuntimeManifestMigrationError(ValueError):
    """Raised when a durable runtime manifest cannot be migrated safely."""


@dataclass(frozen=True, slots=True)
class RuntimeManifestMigrationStep:
    """One explicit schema transition in the durable runtime contract."""

    source_schema: str
    target_schema: str
    migrate: RuntimeManifestMigration

    def __post_init__(self) -> None:
        if not self.source_schema.strip():
            raise ValueError("source_schema must not be empty")
        if not self.target_schema.strip():
            raise ValueError("target_schema must not be empty")
        if self.source_schema == self.target_schema:
            raise ValueError("runtime manifest migration must change schema")
        if not callable(self.migrate):
            raise TypeError("migrate must be callable")


@dataclass(slots=True)
class RuntimeManifestMigrationRegistry:
    """Deterministic one-way migration registry for production runtime manifests.

    The registry intentionally permits only one outgoing edge per source schema.
    Production resume must never guess between competing migration paths.
    """

    _steps: dict[str, RuntimeManifestMigrationStep] = field(default_factory=dict)

    def register(self, step: RuntimeManifestMigrationStep) -> None:
        """Register one transition, rejecting ambiguous or conflicting upgrades."""
        if not isinstance(step, RuntimeManifestMigrationStep):
            raise TypeError("step must be a RuntimeManifestMigrationStep")
        existing = self._steps.get(step.source_schema)
        if existing is not None:
            raise RuntimeManifestMigrationError(
                "runtime manifest source schema already has a migration: "
                f"{step.source_schema} -> {existing.target_schema}"
            )
        self._steps[step.source_schema] = step

    def migrate_payload(
        self,
        payload: Mapping[str, object],
        *,
        target_schema: str = PRODUCTION_RUNTIME_MANIFEST_SCHEMA,
    ) -> dict[str, object]:
        """Migrate a JSON-like payload to ``target_schema`` or fail closed."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        if not target_schema.strip():
            raise ValueError("target_schema must not be empty")

        current: dict[str, object] = dict(payload)
        raw_schema = current.get("schema")
        if not isinstance(raw_schema, str) or not raw_schema.strip():
            raise RuntimeManifestMigrationError(
                "runtime manifest payload has no valid schema"
            )

        visited: set[str] = set()
        while raw_schema != target_schema:
            if raw_schema in visited:
                raise RuntimeManifestMigrationError(
                    f"runtime manifest migration cycle detected at {raw_schema}"
                )
            visited.add(raw_schema)

            step = self._steps.get(raw_schema)
            if step is None:
                raise RuntimeManifestMigrationError(
                    "no runtime manifest migration registered from schema: "
                    + raw_schema
                )

            try:
                migrated = step.migrate(dict(current))
            except Exception as exc:  # migration code is an explicit trust boundary
                raise RuntimeManifestMigrationError(
                    f"runtime manifest migration failed: {step.source_schema} -> "
                    f"{step.target_schema}"
                ) from exc
            if not isinstance(migrated, dict):
                raise RuntimeManifestMigrationError(
                    "runtime manifest migration must return a dict payload"
                )

            produced_schema = migrated.get("schema")
            if produced_schema != step.target_schema:
                raise RuntimeManifestMigrationError(
                    "runtime manifest migration produced unexpected schema: "
                    f"expected {step.target_schema}, got {produced_schema!r}"
                )
            current = dict(migrated)
            raw_schema = step.target_schema

        return current

    def restore(
        self,
        payload: Mapping[str, object],
    ) -> ProductionRuntimeManifest:
        """Migrate to the current contract and run the normal strict validator."""
        migrated = self.migrate_payload(payload)
        try:
            return ProductionRuntimeManifest.restore(migrated)
        except (TypeError, ValueError) as exc:
            raise RuntimeManifestMigrationError(
                "migrated production runtime manifest is invalid"
            ) from exc


__all__ = [
    "RuntimeManifestMigration",
    "RuntimeManifestMigrationError",
    "RuntimeManifestMigrationRegistry",
    "RuntimeManifestMigrationStep",
]
