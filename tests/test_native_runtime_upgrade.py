from __future__ import annotations

import pytest

from cineos.native_video.runtime_manifest import PRODUCTION_RUNTIME_MANIFEST_SCHEMA
from cineos.native_video.runtime_upgrade import (
    RuntimeManifestMigrationError,
    RuntimeManifestMigrationRegistry,
    RuntimeManifestMigrationStep,
)


def _current_payload() -> dict[str, object]:
    return {
        "schema": PRODUCTION_RUNTIME_MANIFEST_SCHEMA,
        "renderer_id": "cineos-native",
        "temporal_model_fingerprint": "temporal-a",
        "device": "cpu",
        "max_recovery_attempts": 2,
        "require_final_film_evaluation": True,
        "require_audio": True,
        "final_gate_policy_fingerprint": "gate-a",
        "native_model_manifest_sha256": "model-a",
    }


def test_current_manifest_restores_without_migration():
    registry = RuntimeManifestMigrationRegistry()

    restored = registry.restore(_current_payload())

    assert restored.renderer_id == "cineos-native"
    assert restored.schema == PRODUCTION_RUNTIME_MANIFEST_SCHEMA


def test_explicit_legacy_migration_is_validated_by_current_restore():
    legacy_schema = "cineos-production-runtime/legacy-test"
    payload = _current_payload()
    payload["schema"] = legacy_schema
    payload["audio_required"] = payload.pop("require_audio")

    def migrate_legacy(raw: dict[str, object]) -> dict[str, object]:
        migrated = dict(raw)
        migrated["schema"] = PRODUCTION_RUNTIME_MANIFEST_SCHEMA
        migrated["require_audio"] = migrated.pop("audio_required")
        return migrated

    registry = RuntimeManifestMigrationRegistry()
    registry.register(
        RuntimeManifestMigrationStep(
            legacy_schema,
            PRODUCTION_RUNTIME_MANIFEST_SCHEMA,
            migrate_legacy,
        )
    )

    restored = registry.restore(payload)

    assert restored.require_audio is True


def test_missing_migration_fails_closed():
    payload = _current_payload()
    payload["schema"] = "cineos-production-runtime/unknown"

    with pytest.raises(
        RuntimeManifestMigrationError, match="no runtime manifest migration"
    ):
        RuntimeManifestMigrationRegistry().restore(payload)


def test_competing_migrations_are_rejected():
    registry = RuntimeManifestMigrationRegistry()
    first = RuntimeManifestMigrationStep(
        "v0", "v1", lambda raw: {**raw, "schema": "v1"}
    )
    second = RuntimeManifestMigrationStep(
        "v0", "v2", lambda raw: {**raw, "schema": "v2"}
    )
    registry.register(first)

    with pytest.raises(RuntimeManifestMigrationError, match="already has a migration"):
        registry.register(second)


def test_migration_must_produce_declared_schema():
    registry = RuntimeManifestMigrationRegistry()
    registry.register(
        RuntimeManifestMigrationStep(
            "v0",
            PRODUCTION_RUNTIME_MANIFEST_SCHEMA,
            lambda raw: {**raw, "schema": "wrong"},
        )
    )

    with pytest.raises(RuntimeManifestMigrationError, match="unexpected schema"):
        registry.migrate_payload({"schema": "v0"})


def test_migration_cycle_fails_closed():
    registry = RuntimeManifestMigrationRegistry()
    registry.register(
        RuntimeManifestMigrationStep("v0", "v1", lambda raw: {**raw, "schema": "v1"})
    )
    registry.register(
        RuntimeManifestMigrationStep("v1", "v0", lambda raw: {**raw, "schema": "v0"})
    )

    with pytest.raises(RuntimeManifestMigrationError, match="cycle detected"):
        registry.migrate_payload({"schema": "v0"})


def test_final_strict_restore_rejects_unknown_fields_after_migration():
    registry = RuntimeManifestMigrationRegistry()
    legacy = _current_payload()
    legacy["schema"] = "v0"
    legacy["optimistic_override"] = True
    registry.register(
        RuntimeManifestMigrationStep(
            "v0",
            PRODUCTION_RUNTIME_MANIFEST_SCHEMA,
            lambda raw: {**raw, "schema": PRODUCTION_RUNTIME_MANIFEST_SCHEMA},
        )
    )

    with pytest.raises(
        RuntimeManifestMigrationError, match="migrated production runtime"
    ):
        registry.restore(legacy)
