from __future__ import annotations

from pathlib import Path

import pytest

from cineos.native_image.model_manifest import (
    ModelManifestError,
    NativeModelComponent,
    NativeModelManifest,
    NativeModelRegistry,
)
from cineos.native_video.production_first_film import (
    build_released_production_first_film_runtime,
)


class _NativeRenderer:
    def render(self, planned, target: str | Path, *, temporal_state):
        raise AssertionError("composition tests must not render video")


def _manifest(version: str, artifact: str) -> NativeModelManifest:
    return NativeModelManifest(
        model_id="cineos-native-video",
        model_version=version,
        runtime_contract_version=1,
        components=(
            NativeModelComponent(
                name="temporal",
                version=version,
                artifact_sha256=artifact,
                contract_version=1,
            ),
        ),
    )


def _registry(tmp_path: Path) -> NativeModelRegistry:
    return NativeModelRegistry(
        path=tmp_path / "registry.json",
        runtime_contract_version=1,
        supported_component_contracts={"temporal": 1},
    )


def _runtime_state(runtime) -> dict[str, object]:
    provider = runtime.runner.orchestrator.checkpoint_state_provider
    assert provider is not None
    state = provider()
    assert state is not None
    return state


def test_released_runtime_binds_active_native_model_manifest(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    active = _manifest("1.0.0", "a" * 64)
    registry.activate(active)

    runtime = build_released_production_first_film_runtime(_NativeRenderer(), registry)

    assert runtime.manifest.native_model_manifest_sha256 == active.manifest_sha256


def test_released_runtime_requires_active_registry_model(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    with pytest.raises(ModelManifestError, match="active native model release"):
        build_released_production_first_film_runtime(_NativeRenderer(), registry)


def test_resume_rejects_changed_native_model_release_before_state_restore(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    first = _manifest("1.0.0", "a" * 64)
    second = _manifest("1.1.0", "b" * 64)
    registry.activate(first)
    saved = build_released_production_first_film_runtime(_NativeRenderer(), registry)
    saved_state = _runtime_state(saved)

    registry.activate(second)
    current = build_released_production_first_film_runtime(_NativeRenderer(), registry)
    before = current.continuity.snapshot()
    restorer = current.runner.orchestrator.checkpoint_state_restorer
    assert restorer is not None

    with pytest.raises(ValueError, match="native_model_manifest_sha256"):
        restorer(saved_state)

    assert current.continuity.snapshot() == before


def test_released_runtime_rejects_conflicting_explicit_manifest_digest(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    registry.activate(_manifest("1.0.0", "a" * 64))

    with pytest.raises(ModelManifestError, match="does not match active registry"):
        build_released_production_first_film_runtime(
            _NativeRenderer(),
            registry,
            native_model_manifest_sha256="f" * 64,
        )
