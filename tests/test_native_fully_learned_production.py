from __future__ import annotations

import hashlib

import pytest

from cineos.native_image.model_manifest import (
    NativeModelComponent,
    NativeModelManifest,
    NativeModelRegistry,
)
from cineos.native_image.neural_backend import NeuralModelConfig, torch_available
from cineos.native_image.neural_decoder import TorchLatentRGBDecoder
from cineos.native_video.film_bridge import temporal_model_fingerprint
from cineos.native_video.fully_learned_production import (
    build_fully_learned_production_first_film_runtime,
)
from cineos.native_video.temporal_deployment import (
    DEFAULT_FULLY_LEARNED_COMPONENT_CONTRACTS,
)
from cineos.native_video.temporal_model import NativeTemporalModel
from cineos.native_video.temporal_model_checkpoint import TemporalModelCheckpoint


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fully_learned_production_shares_temporal_weights(tmp_path) -> None:
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")

    temporal_model = NativeTemporalModel.initialized()
    temporal_checkpoint = TemporalModelCheckpoint.capture(
        temporal_model,
        training_steps=500,
        training_run_id="temporal-prod-001",
        training_data_fingerprint="c" * 64,
    ).save(tmp_path / "temporal.json")

    decoder = TorchLatentRGBDecoder(
        NeuralModelConfig(
            feature_dim=4,
            embedding_dim=8,
            latent_dim=temporal_model.latent_dim,
            hidden_dim=12,
        ),
        width=20,
        height=12,
    ).save_checkpoint(tmp_path / "decoder.pt")

    manifest = NativeModelManifest(
        model_id="cineos-native-video",
        model_version="1.0.0",
        runtime_contract_version=1,
        components=(
            NativeModelComponent(
                name="rgb_decoder",
                version="1.0.0",
                artifact_sha256=_sha256(decoder),
            ),
            NativeModelComponent(
                name="temporal_model",
                version="1.0.0",
                artifact_sha256=_sha256(temporal_checkpoint),
            ),
        ),
    )
    manifest_path = manifest.save(tmp_path / "manifest.json")
    registry = NativeModelRegistry(
        path=tmp_path / "registry.json",
        runtime_contract_version=1,
        supported_component_contracts=dict(
            DEFAULT_FULLY_LEARNED_COMPONENT_CONTRACTS
        ),
    )
    registry.activate(manifest)

    production = build_fully_learned_production_first_film_runtime(
        decoder,
        temporal_checkpoint,
        manifest_path,
        registry,
        fps=6,
    )

    renderer_model = production.renderer_binding.renderer.runtime.model
    continuity_model = production.continuity.model
    assert temporal_model_fingerprint(renderer_model) == temporal_model_fingerprint(
        continuity_model
    )
    assert production.manifest.native_model_manifest_sha256 == manifest.manifest_sha256
    assert production.renderer_binding.renderer.fps == 6
