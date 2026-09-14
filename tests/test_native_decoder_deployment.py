import pytest

from cineos.native_image.neural_backend import NeuralModelConfig, torch_available
from cineos.native_image.neural_decoder import TorchLatentRGBDecoder
from cineos.native_video.deployment import build_checkpoint_temporal_shot_renderer
from cineos.native_video.runtime import NativeTemporalRuntime
from cineos.native_video.temporal_model import NativeTemporalModel


def _save_decoder(tmp_path, *, latent_dim: int):
    decoder = TorchLatentRGBDecoder(
        NeuralModelConfig(
            feature_dim=4,
            embedding_dim=8,
            latent_dim=latent_dim,
            hidden_dim=12,
        ),
        width=20,
        height=12,
    )
    return decoder.save_checkpoint(tmp_path / f"decoder-{latent_dim}.pt")


def test_deployment_builder_uses_checkpoint_resolution_and_weights(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    runtime = NativeTemporalRuntime.default()
    checkpoint = _save_decoder(tmp_path, latent_dim=runtime.model.latent_dim)

    renderer = build_checkpoint_temporal_shot_renderer(
        checkpoint,
        runtime=runtime,
        fps=6,
    )

    assert renderer.runtime is runtime
    assert renderer.width == 20
    assert renderer.height == 12
    assert renderer.fps == 6
    assert renderer.decoder.decoder_id.startswith(
        "cineos-torch-rgb-decoder/0.1@sha256:"
    )


def test_deployment_builder_rejects_incompatible_temporal_latent_dim(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    model = NativeTemporalModel.initialized(latent_dim=7)
    runtime = NativeTemporalRuntime.default(model=model)
    checkpoint = _save_decoder(tmp_path, latent_dim=6)

    with pytest.raises(ValueError, match="incompatible with temporal model"):
        build_checkpoint_temporal_shot_renderer(checkpoint, runtime=runtime)
