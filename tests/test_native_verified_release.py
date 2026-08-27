"""Regression tests for cryptographically verified production model releases."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cineos.native_image.artifact_verification import ModelArtifactVerificationError
from cineos.native_image.model_manifest import (
    NativeModelComponent,
    NativeModelManifest,
    NativeModelRegistry,
)
from cineos.native_video.verified_release import (
    build_verified_released_production_first_film_runtime,
)


class _NativeRenderer:
    def render(self, planned, target: str | Path, *, temporal_state):
        raise AssertionError("composition tests must not render video")


def _registry(tmp_path: Path) -> NativeModelRegistry:
    return NativeModelRegistry(
        path=tmp_path / "registry.json",
        runtime_contract_version=1,
        supported_component_contracts={"temporal": 1},
    )


def _activate_for_file(
    registry: NativeModelRegistry,
    artifact: Path,
) -> NativeModelManifest:
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = NativeModelManifest(
        model_id="cineos-native-video",
        model_version="1.0.0",
        runtime_contract_version=1,
        components=(
            NativeModelComponent(
                name="temporal",
                version="1.0.0",
                artifact_sha256=digest,
                contract_version=1,
            ),
        ),
    )
    registry.activate(manifest)
    return manifest


def test_verified_release_binds_runtime_to_measured_model_bytes(tmp_path: Path) -> None:
    weights = tmp_path / "temporal.bin"
    weights.write_bytes(b"cineos-temporal-weights-v1")
    registry = _registry(tmp_path)
    manifest = _activate_for_file(registry, weights)

    verified = build_verified_released_production_first_film_runtime(
        _NativeRenderer(),
        registry,
        {"temporal": weights},
    )

    assert (
        verified.runtime.manifest.native_model_manifest_sha256
        == manifest.manifest_sha256
    )
    assert verified.model_artifacts.manifest_sha256 == manifest.manifest_sha256
    assert len(verified.model_artifacts.components) == 1
    component = verified.model_artifacts.components[0]
    assert component.name == "temporal"
    assert component.actual_sha256 == component.expected_sha256
    assert component.size_bytes == weights.stat().st_size
    assert len(verified.model_artifacts.fingerprint) == 64


def test_verified_release_rejects_tampered_model_bytes(tmp_path: Path) -> None:
    weights = tmp_path / "temporal.bin"
    weights.write_bytes(b"approved-weights")
    registry = _registry(tmp_path)
    _activate_for_file(registry, weights)
    weights.write_bytes(b"tampered-after-activation")

    with pytest.raises(ModelArtifactVerificationError, match="SHA-256 mismatch"):
        build_verified_released_production_first_film_runtime(
            _NativeRenderer(),
            registry,
            {"temporal": weights},
        )


def test_verified_release_rejects_missing_component_file(tmp_path: Path) -> None:
    weights = tmp_path / "temporal.bin"
    weights.write_bytes(b"approved-weights")
    registry = _registry(tmp_path)
    _activate_for_file(registry, weights)

    with pytest.raises(ModelArtifactVerificationError, match="missing model component"):
        build_verified_released_production_first_film_runtime(
            _NativeRenderer(),
            registry,
            {},
        )


def test_verified_release_rejects_unversioned_extra_component(tmp_path: Path) -> None:
    weights = tmp_path / "temporal.bin"
    weights.write_bytes(b"approved-weights")
    extra = tmp_path / "shadow.bin"
    extra.write_bytes(b"unversioned-shadow-model")
    registry = _registry(tmp_path)
    _activate_for_file(registry, weights)

    with pytest.raises(
        ModelArtifactVerificationError, match="unexpected model component"
    ):
        build_verified_released_production_first_film_runtime(
            _NativeRenderer(),
            registry,
            {"temporal": weights, "shadow": extra},
        )
