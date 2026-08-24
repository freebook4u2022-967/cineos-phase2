from __future__ import annotations

import json

import pytest

from cineos.native_image.model_manifest import (
    ModelManifestError,
    NativeModelComponent,
    NativeModelManifest,
    NativeModelRegistry,
    check_runtime_compatibility,
)


def _sha(byte: str) -> str:
    return byte * 64


def _manifest(version: str, artifact: str = "a", contract: int = 1) -> NativeModelManifest:
    return NativeModelManifest(
        model_id="cineos-native-film",
        model_version=version,
        runtime_contract_version=1,
        components=(
            NativeModelComponent(
                name="frame-model",
                version=version,
                artifact_sha256=_sha(artifact),
                contract_version=contract,
            ),
        ),
        metadata={"track": "production-candidate"},
    )


def test_manifest_hash_is_deterministic_and_round_trips(tmp_path):
    manifest = _manifest("1.2.3")
    path = manifest.save(tmp_path / "model.json")

    loaded = NativeModelManifest.load(path)

    assert loaded == manifest
    assert loaded.manifest_sha256 == manifest.manifest_sha256
    assert json.loads(path.read_text())["manifest_sha256"] == manifest.manifest_sha256


def test_manifest_rejects_tampering(tmp_path):
    path = _manifest("1.2.3").save(tmp_path / "model.json")
    payload = json.loads(path.read_text())
    payload["model_version"] = "9.9.9"
    path.write_text(json.dumps(payload))

    with pytest.raises(ModelManifestError, match="hash mismatch"):
        NativeModelManifest.load(path)


def test_runtime_compatibility_fails_closed_for_newer_component_contract():
    result = check_runtime_compatibility(
        _manifest("1.0.0", contract=2),
        runtime_contract_version=1,
        supported_component_contracts={"frame-model": 1},
    )

    assert result.compatible is False
    assert "requires contract 2" in result.reason


def test_registry_activation_and_rollback_are_persistent(tmp_path):
    path = tmp_path / "registry.json"
    registry = NativeModelRegistry(
        path=path,
        runtime_contract_version=1,
        supported_component_contracts={"frame-model": 1},
    )
    first = _manifest("1.0.0", artifact="a")
    second = _manifest("1.1.0", artifact="b")

    registry.activate(first)
    registry.activate(second)
    assert registry.active() == second

    reloaded = NativeModelRegistry(
        path=path,
        runtime_contract_version=1,
        supported_component_contracts={"frame-model": 1},
    )
    assert reloaded.rollback() == first
    assert reloaded.active() == first


def test_registry_refuses_incompatible_activation(tmp_path):
    registry = NativeModelRegistry(
        path=tmp_path / "registry.json",
        runtime_contract_version=1,
        supported_component_contracts={"frame-model": 1},
    )

    with pytest.raises(ModelManifestError, match="refusing incompatible model activation"):
        registry.activate(_manifest("2.0.0", contract=2))
