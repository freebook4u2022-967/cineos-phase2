from pathlib import Path

import pytest

from cineos.native_image.neural_backend import NeuralModelConfig
from cineos.native_image.neural_data import ApprovedManifestPreprocessor
from cineos.native_image.training import NativeDatasetManifest, NativeTrainingSample


def _config() -> NeuralModelConfig:
    return NeuralModelConfig(
        feature_dim=4,
        embedding_dim=8,
        latent_dim=6,
        hidden_dim=12,
    )


def _sample() -> NativeTrainingSample:
    return NativeTrainingSample(
        sample_id="approved-001",
        image_path="frames/shot.ppm",
        character_reference_paths=("references/hero.ppm",),
        caption="Hero turns toward camera",
        scene_description="Rainy port at night",
        identity_tags=("hero", "approved-face"),
        continuity_tags=("same-wardrobe",),
    )


def test_approved_manifest_preprocessor_reads_only_declared_dataset_files(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "references").mkdir()
    (tmp_path / "frames" / "shot.ppm").write_bytes(b"P6 frame bytes")
    (tmp_path / "references" / "hero.ppm").write_bytes(b"P6 hero bytes")
    prepared = ApprovedManifestPreprocessor(_config(), tmp_path).prepare(_sample())

    assert prepared.sample_id == "approved-001"
    assert len(prepared.identity_features) == 4
    assert len(prepared.scene_features) == 4
    assert len(prepared.target_latent) == 6


def test_preprocessor_rejects_paths_outside_dataset_root(tmp_path):
    outside = tmp_path.parent / "outside.ppm"
    outside.write_bytes(b"outside")
    sample = NativeTrainingSample(
        sample_id="bad",
        image_path=f"../{outside.name}",
        character_reference_paths=("refs/hero.ppm",),
        caption="bad path",
        scene_description="",
    )
    with pytest.raises(ValueError, match="escapes configured dataset root"):
        ApprovedManifestPreprocessor(_config(), tmp_path).prepare(sample)


def test_manifest_preparation_is_deterministic(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "references").mkdir()
    (tmp_path / "frames" / "shot.ppm").write_bytes(b"frame")
    (tmp_path / "references" / "hero.ppm").write_bytes(b"hero")
    manifest = NativeDatasetManifest("demo", "1", [_sample()])
    preprocessor = ApprovedManifestPreprocessor(_config(), Path(tmp_path))

    first = preprocessor.prepare_dataset(manifest)
    second = preprocessor.prepare_dataset(manifest)
    assert first == second
