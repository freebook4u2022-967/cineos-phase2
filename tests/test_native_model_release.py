from __future__ import annotations

import json
from pathlib import Path

import pytest

from cineos.native_image.checkpoint_gate import CheckpointBenchmarkGate, CheckpointScore
from cineos.native_image.model_manifest import (
    ModelManifestError,
    NativeModelComponent,
    NativeModelManifest,
    NativeModelRegistry,
)
from cineos.native_image.model_release import NativeModelReleaseController


def _component(name: str, *, contract_version: int = 1) -> NativeModelComponent:
    return NativeModelComponent(
        name=name,
        version="1.0.0",
        artifact_sha256="a" * 64,
        contract_version=contract_version,
    )


def _manifest(version: str, *, contract_version: int = 1) -> NativeModelManifest:
    return NativeModelManifest(
        model_id="cineos-native-video",
        model_version=version,
        runtime_contract_version=1,
        components=(_component("temporal", contract_version=contract_version),),
    )


def _score(checkpoint_id: str, identity: float) -> CheckpointScore:
    return CheckpointScore(
        checkpoint_id=checkpoint_id,
        reconstruction_mse=0.10,
        same_character_scene_distance=0.80,
        different_character_same_scene_distance=0.80,
        identity_consistency_score=identity,
    )


def _controller(tmp_path: Path) -> NativeModelReleaseController:
    registry = NativeModelRegistry(
        path=tmp_path / "registry.json",
        runtime_contract_version=1,
        supported_component_contracts={"temporal": 1},
    )
    return NativeModelReleaseController(
        registry=registry,
        benchmark_gate=CheckpointBenchmarkGate(minimum_improvement=0.01),
        release_record_path=tmp_path / "last_release.json",
    )


def test_promotes_compatible_candidate_that_beats_incumbent(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    incumbent_manifest = _manifest("1.0.0")
    controller.registry.activate(incumbent_manifest)

    decision = controller.promote(
        _manifest("1.1.0"),
        _score("candidate", 0.95),
        _score("incumbent", 0.75),
    )

    assert decision.promoted is True
    active = controller.registry.active()
    assert active is not None
    assert active.model_version == "1.1.0"
    record = json.loads((tmp_path / "last_release.json").read_text(encoding="utf-8"))
    assert record["promoted"] is True
    assert record["previous_manifest_sha256"] == incumbent_manifest.manifest_sha256


def test_quality_rejection_does_not_mutate_active_model(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    incumbent_manifest = _manifest("1.0.0")
    controller.registry.activate(incumbent_manifest)

    decision = controller.promote(
        _manifest("1.1.0"),
        _score("candidate", 0.76),
        _score("incumbent", 0.75),
    )

    assert decision.promoted is False
    assert "benchmark gate rejected" in decision.reason
    active = controller.registry.active()
    assert active is not None
    assert active.manifest_sha256 == incumbent_manifest.manifest_sha256


def test_incompatible_candidate_is_rejected_before_activation(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    incumbent_manifest = _manifest("1.0.0")
    controller.registry.activate(incumbent_manifest)

    decision = controller.promote(
        _manifest("2.0.0", contract_version=2),
        _score("candidate", 0.99),
        _score("incumbent", 0.75),
    )

    assert decision.promoted is False
    assert "runtime compatibility rejected" in decision.reason
    active = controller.registry.active()
    assert active is not None
    assert active.manifest_sha256 == incumbent_manifest.manifest_sha256


def test_promoted_model_can_rollback_to_previous_compatible_model(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    incumbent_manifest = _manifest("1.0.0")
    controller.registry.activate(incumbent_manifest)
    controller.promote(
        _manifest("1.1.0"),
        _score("candidate", 0.95),
        _score("incumbent", 0.75),
    )

    rolled_back = controller.rollback()

    assert rolled_back.manifest_sha256 == incumbent_manifest.manifest_sha256
    active = controller.registry.active()
    assert active is not None
    assert active.manifest_sha256 == incumbent_manifest.manifest_sha256


def test_registry_still_fails_closed_if_runtime_changes_between_eval_and_activate(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    candidate = _manifest("1.1.0")
    decision = controller.evaluate(candidate, _score("candidate", 0.95), None)
    assert decision.promoted is True

    controller.registry.supported_component_contracts = {}
    with pytest.raises(
        ModelManifestError, match="refusing incompatible model activation"
    ):
        controller.registry.activate(candidate)
