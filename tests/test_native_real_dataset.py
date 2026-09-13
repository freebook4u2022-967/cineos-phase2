import pytest

from cineos.native_image.neural_backend import NeuralModelConfig, torch_available
from cineos.native_image.real_dataset import (
    RealManifestTorchDataset,
    build_distributed_real_loader,
)
from cineos.native_image.training import NativeDatasetManifest, NativeTrainingSample


def _manifest():
    return NativeDatasetManifest(
        "real-demo",
        "1",
        [
            NativeTrainingSample(
                sample_id="sample-1",
                image_path="frames/one.ppm",
                character_reference_paths=("refs/hero.ppm",),
                caption="Hero looks left",
                scene_description="Night street",
                identity_tags=("hero",),
                continuity_tags=("same-coat",),
            ),
            NativeTrainingSample(
                sample_id="sample-2",
                image_path="frames/two.ppm",
                character_reference_paths=("refs/hero.ppm",),
                caption="Hero looks right",
                scene_description="Night street",
                identity_tags=("hero",),
                continuity_tags=("same-coat",),
            ),
        ],
    )


def _prepare_files(root):
    (root / "frames").mkdir()
    (root / "refs").mkdir()
    (root / "frames" / "one.ppm").write_bytes(b"frame-one")
    (root / "frames" / "two.ppm").write_bytes(b"frame-two")
    (root / "refs" / "hero.ppm").write_bytes(b"hero-ref")


def test_real_manifest_dataset_produces_training_tensors(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    _prepare_files(tmp_path)
    config = NeuralModelConfig(
        feature_dim=4, embedding_dim=8, latent_dim=6, hidden_dim=12
    )
    dataset = RealManifestTorchDataset(_manifest(), tmp_path, config)
    identity, scene, source, target, sample_id = dataset[0]
    assert identity.shape == (4,)
    assert scene.shape == (4,)
    assert source.shape == (6,)
    assert target.shape == (6,)
    assert sample_id == "sample-1"


def test_distributed_loader_partitions_real_manifest_by_rank(tmp_path):
    if not torch_available():
        pytest.skip("PyTorch optional dependency is not installed")
    _prepare_files(tmp_path)
    config = NeuralModelConfig(
        feature_dim=4, embedding_dim=8, latent_dim=6, hidden_dim=12
    )
    dataset = RealManifestTorchDataset(_manifest(), tmp_path, config)
    _, sampler0 = build_distributed_real_loader(
        dataset, rank=0, world_size=2, batch_size=1, shuffle=False
    )
    _, sampler1 = build_distributed_real_loader(
        dataset, rank=1, world_size=2, batch_size=1, shuffle=False
    )
    assert list(iter(sampler0)) == [0]
    assert list(iter(sampler1)) == [1]
