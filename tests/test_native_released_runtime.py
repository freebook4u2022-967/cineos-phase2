"""Regression tests for strict released native-video composition."""

from __future__ import annotations

from pathlib import Path

import pytest

from cineos.native_image.model_manifest import (
    NativeModelComponent,
    NativeModelManifest,
    NativeModelRegistry,
)
from cineos.native_video.final_gate import MeasuredFinalFilmGate
from cineos.native_video.released_runtime import (
    build_strict_released_production_runtime,
)


class _NativeRenderer:
    def render(self, planned, target: str | Path, *, temporal_state):
        raise AssertionError("composition tests must not render video")


def _registry(tmp_path: Path) -> NativeModelRegistry:
    registry = NativeModelRegistry(
        path=tmp_path / "registry.json",
        runtime_contract_version=1,
        supported_component_contracts={"temporal": 1},
    )
    registry.activate(
        NativeModelManifest(
            model_id="cineos-native-video",
            model_version="1.0.0",
            runtime_contract_version=1,
            components=(
                NativeModelComponent(
                    name="temporal",
                    version="1.0.0",
                    artifact_sha256="a" * 64,
                    contract_version=1,
                ),
            ),
        )
    )
    return registry


def test_strict_released_runtime_enables_audio_qc_by_default(tmp_path: Path) -> None:
    runtime = build_strict_released_production_runtime(
        _NativeRenderer(),
        _registry(tmp_path),
    )

    assert runtime.final_gate.require_audio is True
    assert runtime.final_gate.audio_evaluator is not None
    assert runtime.manifest.require_audio is True


def test_strict_released_runtime_rejects_audio_qc_downgrade(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires measured final-film audio QC"):
        build_strict_released_production_runtime(
            _NativeRenderer(),
            _registry(tmp_path),
            final_gate=MeasuredFinalFilmGate(require_audio=False),
        )


def test_strict_released_runtime_accepts_custom_strict_gate(tmp_path: Path) -> None:
    gate = MeasuredFinalFilmGate(require_audio=True)

    runtime = build_strict_released_production_runtime(
        _NativeRenderer(),
        _registry(tmp_path),
        final_gate=gate,
    )

    assert runtime.final_gate is gate
    assert runtime.manifest.require_audio is True


def test_strict_released_runtime_rejects_non_measured_gate(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be MeasuredFinalFilmGate"):
        build_strict_released_production_runtime(
            _NativeRenderer(),
            _registry(tmp_path),
            final_gate=object(),
        )
