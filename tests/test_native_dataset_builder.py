from cineos.native_image.dataset_builder import RealTrainingDatasetBuilder
from cineos.native_image.training import NativeTrainingSample


def _sample(sample_id, image, *, identity=("arif",), continuity=("scene-1",)):
    return NativeTrainingSample(
        sample_id=sample_id,
        image_path=image,
        character_reference_paths=("refs/arif.ppm",),
        caption="Arif in a cinematic scene",
        scene_description="night interior",
        identity_tags=identity,
        continuity_tags=continuity,
    )


def test_builder_accepts_valid_sample(tmp_path):
    (tmp_path / "refs").mkdir()
    (tmp_path / "frames").mkdir()
    (tmp_path / "refs/arif.ppm").write_bytes(b"reference-image-data")
    (tmp_path / "frames/a.ppm").write_bytes(b"unique-training-image-a")
    result = RealTrainingDatasetBuilder(tmp_path).build(
        "set", "1", [_sample("a", "frames/a.ppm")]
    )
    assert len(result.manifest.samples) == 1
    assert result.rejected == ()


def test_builder_rejects_duplicate_image_content(tmp_path):
    (tmp_path / "refs").mkdir()
    (tmp_path / "frames").mkdir()
    (tmp_path / "refs/arif.ppm").write_bytes(b"reference-image-data")
    payload = b"same-training-image-content"
    (tmp_path / "frames/a.ppm").write_bytes(payload)
    (tmp_path / "frames/b.ppm").write_bytes(payload)
    result = RealTrainingDatasetBuilder(tmp_path).build(
        "set", "1", [_sample("a", "frames/a.ppm"), _sample("b", "frames/b.ppm")]
    )
    assert len(result.manifest.samples) == 1
    assert result.rejected[0].reason == "duplicate training image"


def test_builder_rejects_missing_continuity_metadata(tmp_path):
    (tmp_path / "refs").mkdir()
    (tmp_path / "frames").mkdir()
    (tmp_path / "refs/arif.ppm").write_bytes(b"reference-image-data")
    (tmp_path / "frames/a.ppm").write_bytes(b"unique-training-image-a")
    result = RealTrainingDatasetBuilder(tmp_path).build(
        "set", "1", [_sample("a", "frames/a.ppm", continuity=())]
    )
    assert result.manifest.samples == []
    assert result.rejected[0].reason == "continuity metadata missing"


def test_builder_rejects_missing_character_reference(tmp_path):
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames/a.ppm").write_bytes(b"unique-training-image-a")
    result = RealTrainingDatasetBuilder(tmp_path).build(
        "set", "1", [_sample("a", "frames/a.ppm")]
    )
    assert result.manifest.samples == []
    assert result.rejected[0].reason == "character reference missing"
