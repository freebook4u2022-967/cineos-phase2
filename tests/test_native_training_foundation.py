import json

from cineos.native_image.training import (
    NativeCheckpointManifest,
    NativeDatasetManifest,
    NativeTrainingSample,
)


def _sample(sample_id: str = "sample-001") -> NativeTrainingSample:
    return NativeTrainingSample(
        sample_id=sample_id,
        image_path=f"frames/{sample_id}.png",
        character_reference_paths=("refs/hero-front.png", "refs/hero-full.png"),
        caption="A close-up of the hero receiving a mysterious message.",
        scene_description="Night interior, practical lamp, suspenseful mood.",
        identity_tags=("hero", "same-face"),
        continuity_tags=("dark-jacket", "phone-in-right-hand"),
    )


def test_dataset_manifest_is_deterministic_and_persistable(tmp_path):
    manifest = NativeDatasetManifest(name="cineos-short-drama", version="0.1")
    manifest.add(_sample())

    first_hash = manifest.content_hash()
    assert len(first_hash) == 64
    assert first_hash == manifest.content_hash()

    destination = manifest.save(tmp_path / "dataset.json")
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 1
    assert payload["content_hash"] == first_hash


def test_dataset_manifest_rejects_duplicate_sample_ids():
    manifest = NativeDatasetManifest(name="cineos-short-drama", version="0.1")
    manifest.add(_sample())
    try:
        manifest.add(_sample())
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate training sample was accepted")


def test_checkpoint_manifest_requires_all_learned_components(tmp_path):
    checkpoint = NativeCheckpointManifest(
        model_name="cineos-frame-model",
        model_version="0.1",
        dataset_hash="a" * 64,
        training_step=100,
        component_files={
            "identity_encoder": "identity.safetensors",
            "scene_encoder": "scene.safetensors",
            "sampler": "sampler.safetensors",
            "decoder": "decoder.safetensors",
        },
        metrics={"validation_loss": 0.25},
    )
    path = checkpoint.save(tmp_path / "checkpoint.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["training_step"] == 100
    assert payload["component_files"]["sampler"] == "sampler.safetensors"
