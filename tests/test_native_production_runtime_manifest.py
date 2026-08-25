from __future__ import annotations

import pytest

from cineos.native_video import (
    PRODUCTION_RUNTIME_MANIFEST_SCHEMA,
    ProductionRuntimeManifest,
)


def _manifest(**overrides: object) -> ProductionRuntimeManifest:
    values: dict[str, object] = {
        "renderer_id": "cineos-native",
        "temporal_model_fingerprint": "abc123",
        "device": "cpu",
        "max_recovery_attempts": 2,
        "require_final_film_evaluation": True,
        "require_audio": True,
    }
    values.update(overrides)
    return ProductionRuntimeManifest(**values)  # type: ignore[arg-type]


def test_manifest_snapshot_round_trip() -> None:
    manifest = _manifest()

    restored = ProductionRuntimeManifest.restore(manifest.snapshot())

    assert restored == manifest
    assert restored.schema == PRODUCTION_RUNTIME_MANIFEST_SCHEMA


def test_manifest_rejects_unknown_schema() -> None:
    payload = _manifest().snapshot()
    payload["schema"] = "cineos-production-runtime/99"

    with pytest.raises(
        ValueError, match="unsupported production runtime manifest schema"
    ):
        ProductionRuntimeManifest.restore(payload)


def test_resume_compatibility_allows_recovery_budget_change() -> None:
    current = _manifest(max_recovery_attempts=5)
    saved = _manifest(max_recovery_attempts=1)

    current.assert_resume_compatible(saved)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("renderer_id", "different-renderer"),
        ("temporal_model_fingerprint", "different-model"),
        ("device", "cuda"),
        ("require_final_film_evaluation", False),
        ("require_audio", False),
    ],
)
def test_resume_compatibility_rejects_changed_production_invariants(
    field_name: str,
    value: object,
) -> None:
    current = _manifest()
    saved = _manifest(**{field_name: value})

    with pytest.raises(ValueError, match=field_name):
        current.assert_resume_compatible(saved)


def test_manifest_rejects_empty_runtime_identity() -> None:
    with pytest.raises(ValueError, match="renderer_id"):
        _manifest(renderer_id=" ")

    with pytest.raises(ValueError, match="temporal_model_fingerprint"):
        _manifest(temporal_model_fingerprint="")
