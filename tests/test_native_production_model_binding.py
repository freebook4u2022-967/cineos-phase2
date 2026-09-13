"""Regression tests for durable production native-model release binding."""

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
    def __init__(self, native_model_manifest_sha256: str | None = None) -> None:
        if native_model_manifest_sha256 is not None:
            self.native_model_manifest_sha256 = native_model_manifest_sha256

    def render(self, planned, target: str | Path, *, temporal_state):
        raise AssertionError("composition tests must not render video")


class _MethodBoundNativeRenderer:
    def __init__(self, digest: str) -> None:
        self._digest = digest

    def native_model_manifest_sha256(self) -> str:
        return self._digest

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


def _renderer_for(manifest: NativeModelManifest) -> _NativeRenderer:
    return _NativeRenderer(manifest.manifest_sha256)


def test_released_runtime_binds_active_native_model_manifest(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    active = _manifest("1.0.0", "a" * 64)
    registry.activate(active)

    runtime = build_released_production_first_film_runtime(
        _renderer_for(active), registry
    )

    assert runtime.manifest.native_model_manifest_sha256 == active.manifest_sha256


def test_released_runtime_accepts_method_reported_renderer_provenance(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    active = _manifest("1.0.0", "a" * 64)
    registry.activate(active)

    runtime = build_released_production_first_film_runtime(
        _MethodBoundNativeRenderer(active.manifest_sha256.upper()), registry
    )

    assert runtime.manifest.native_model_manifest_sha256 == active.manifest_sha256


def test_released_runtime_requires_active_registry_model(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    with pytest.raises(ModelManifestError, match="active native model release"):
        build_released_production_first_film_runtime(
            _NativeRenderer("a" * 64), registry
        )


def test_released_runtime_requires_renderer_loaded_release_provenance(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    active = _manifest("1.0.0", "a" * 64)
    registry.activate(active)

    with pytest.raises(ModelManifestError, match="must expose"):
        build_released_production_first_film_runtime(_NativeRenderer(), registry)


def test_released_runtime_rejects_mismatched_renderer_loaded_release(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    active = _manifest("1.0.0", "a" * 64)
    registry.activate(active)

    with pytest.raises(ModelManifestError, match="does not match active registry"):
        build_released_production_first_film_runtime(
            _NativeRenderer("f" * 64), registry
        )


def test_released_runtime_rejects_malformed_renderer_release_digest(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    active = _manifest("1.0.0", "a" * 64)
    registry.activate(active)

    with pytest.raises(ModelManifestError, match="64-character hex digest"):
        build_released_production_first_film_runtime(
            _NativeRenderer("not-a-digest"), registry
        )


def test_resume_rejects_changed_native_model_release_before_state_restore(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    first = _manifest("1.0.0", "a" * 64)
    second = _manifest("1.1.0", "b" * 64)
    registry.activate(first)
    saved = build_released_production_first_film_runtime(_renderer_for(first), registry)
    saved_state = _runtime_state(saved)

    registry.activate(second)
    current = build_released_production_first_film_runtime(
        _renderer_for(second), registry
    )
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
    active = _manifest("1.0.0", "a" * 64)
    registry.activate(active)

    with pytest.raises(ModelManifestError, match="does not match active registry"):
        build_released_production_first_film_runtime(
            _renderer_for(active),
            registry,
            native_model_manifest_sha256="f" * 64,
        )


def test_released_runtime_rechecks_compatibility_after_runtime_upgrade(
    tmp_path: Path,
) -> None:
    """A persisted active model must not bypass a newer runtime contract gate."""
    registry = _registry(tmp_path)
    active = _manifest("1.0.0", "a" * 64)
    registry.activate(active)

    upgraded_runtime = NativeModelRegistry(
        path=registry.path,
        runtime_contract_version=1,
        supported_component_contracts={"other-component": 1},
    )

    error = (
        "refusing incompatible active native model release: "
        "unsupported component: temporal"
    )
    with pytest.raises(ModelManifestError, match=error):
        build_released_production_first_film_runtime(
            _renderer_for(active), upgraded_runtime
        )
