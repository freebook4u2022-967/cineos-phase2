from __future__ import annotations

import hashlib

import pytest

from cineos.native_image.model_manifest import (
    ModelManifestError,
    NativeModelComponent,
    NativeModelManifest,
)
from cineos.native_video.temporal_deployment import (
    DEFAULT_TEMPORAL_COMPONENT_NAME,
    validate_fully_learned_native_video_release,
)
from cineos.native_video.temporal_model import NativeTemporalModel
from cineos.native_video.temporal_model_checkpoint import TemporalModelCheckpoint


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release(tmp_path):
    decoder = tmp_path / "decoder.pt"
    decoder.write_bytes(b"real-decoder-artifact")
    temporal = TemporalModelCheckpoint.capture(
        NativeTemporalModel.initialized(),
        training_steps=100,
        training_run_id="temporal-train-001",
        training_data_fingerprint="b" * 64,
    ).save(tmp_path / "temporal.json")
    manifest = NativeModelManifest(
        model_id="cineos-native-video",
        model_version="1.0.0",
        runtime_contract_version=1,
        components=(
            NativeModelComponent(
                name="rgb_decoder",
                version="1.0.0",
                artifact_sha256=_sha256(decoder),
                contract_version=1,
            ),
            NativeModelComponent(
                name=DEFAULT_TEMPORAL_COMPONENT_NAME,
                version="1.0.0",
                artifact_sha256=_sha256(temporal),
                contract_version=1,
            ),
        ),
    )
    manifest_path = manifest.save(tmp_path / "manifest.json")
    return decoder, temporal, manifest_path


def test_fully_learned_release_binds_decoder_and_temporal_artifacts(tmp_path) -> None:
    decoder, temporal, manifest_path = _release(tmp_path)

    manifest, checkpoint = validate_fully_learned_native_video_release(
        decoder,
        temporal,
        manifest_path,
    )

    assert manifest.model_version == "1.0.0"
    assert checkpoint.training_steps == 100
    assert checkpoint.training_run_id == "temporal-train-001"


def test_fully_learned_release_rejects_temporal_artifact_drift(tmp_path) -> None:
    decoder, temporal, manifest_path = _release(tmp_path)
    temporal.write_text(temporal.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ModelManifestError, match="temporal_model"):
        validate_fully_learned_native_video_release(
            decoder,
            temporal,
            manifest_path,
        )


def test_fully_learned_release_rejects_decoder_artifact_drift(tmp_path) -> None:
    decoder, temporal, manifest_path = _release(tmp_path)
    decoder.write_bytes(b"different-decoder")

    with pytest.raises(ModelManifestError, match="rgb_decoder"):
        validate_fully_learned_native_video_release(
            decoder,
            temporal,
            manifest_path,
        )


def test_fully_learned_release_rejects_manifest_without_temporal_component(
    tmp_path,
) -> None:
    decoder, temporal, _ = _release(tmp_path)
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
        ),
    )
    manifest_path = manifest.save(tmp_path / "decoder-only-manifest.json")

    with pytest.raises(ModelManifestError, match="temporal_model"):
        validate_fully_learned_native_video_release(
            decoder,
            temporal,
            manifest_path,
        )


def test_fully_learned_release_rejects_newer_temporal_contract(tmp_path) -> None:
    decoder, temporal, _ = _release(tmp_path)
    manifest = NativeModelManifest(
        model_id="cineos-native-video",
        model_version="1.1.0",
        runtime_contract_version=1,
        components=(
            NativeModelComponent(
                name="rgb_decoder",
                version="1.0.0",
                artifact_sha256=_sha256(decoder),
            ),
            NativeModelComponent(
                name=DEFAULT_TEMPORAL_COMPONENT_NAME,
                version="2.0.0",
                artifact_sha256=_sha256(temporal),
                contract_version=2,
            ),
        ),
    )
    manifest_path = manifest.save(tmp_path / "future-manifest.json")

    with pytest.raises(ModelManifestError, match="requires contract 2"):
        validate_fully_learned_native_video_release(
            decoder,
            temporal,
            manifest_path,
        )
