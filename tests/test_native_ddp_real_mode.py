import json

import pytest

from cineos.native_image.ddp_entrypoint import load_dataset_manifest


def test_load_dataset_manifest_restores_training_samples(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "cineos-native-training-dataset/0.1",
                "name": "real-set",
                "version": "1",
                "samples": [
                    {
                        "sample_id": "shot-001",
                        "image_path": "frames/001.ppm",
                        "character_reference_paths": ["refs/arif.ppm"],
                        "caption": "Arif enters the room",
                        "scene_description": "dark cinematic room",
                        "identity_tags": ["arif"],
                        "continuity_tags": ["scene-1"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = load_dataset_manifest(path)
    assert manifest.name == "real-set"
    assert manifest.samples[0].sample_id == "shot-001"
    assert manifest.samples[0].identity_tags == ("arif",)


def test_load_dataset_manifest_rejects_invalid_sample(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "name": "bad",
                "version": "1",
                "samples": [
                    {
                        "sample_id": "shot-001",
                        "image_path": "frames/001.ppm",
                        "character_reference_paths": [],
                        "caption": "bad",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="character references"):
        load_dataset_manifest(path)
