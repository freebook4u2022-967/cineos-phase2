from __future__ import annotations

import hashlib

import pytest

from cineos.native_image.model_manifest import (
    ModelManifestError,
    NativeModelComponent,
    NativeModelManifest,
)
from cineos.native_video.deployment import (
    DEFAULT_DECODER_COMPONENT_NAME,
    validate_checkpoint_against_native_model_manifest,
)


def _manifest_for(checkpoint_bytes: bytes, *, contract_version: int = 1) -> NativeModelManifest:
    return NativeModelManifest(
        model_id="cineos-native-video-test",
        model_version="0.1.0",
        runtime_contract_version=1,
        components=(
            NativeModelComponent(
                name=DEFAULT_DECODER_COMPONENT_NAME,
                version="0.1.0",
                artifact_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
                contract_version=contract_version,
            ),
        ),
    )


def test_validate_checkpoint_against_manifest_accepts_exact_released_bytes(tmp_path):
    checkpoint_bytes = b"cineos-trained-decoder-checkpoint"
    checkpoint = tmp_path / "decoder.pt"
    checkpoint.write_bytes(checkpoint_bytes)
    manifest = _manifest_for(checkpoint_bytes)
    manifest_path = manifest.save(tmp_path / "manifest.json")

    verified = validate_checkpoint_against_native_model_manifest(
        checkpoint,
        manifest_path,
    )

    assert verified.manifest_sha256 == manifest.manifest_sha256


def test_validate_checkpoint_against_manifest_rejects_tampered_checkpoint(tmp_path):
    released = b"released-checkpoint"
    checkpoint = tmp_path / "decoder.pt"
    checkpoint.write_bytes(released)
    manifest_path = _manifest_for(released).save(tmp_path / "manifest.json")
    checkpoint.write_bytes(released + b"-tampered")

    with pytest.raises(ModelManifestError, match="checkpoint digest does not match"):
        validate_checkpoint_against_native_model_manifest(checkpoint, manifest_path)


def test_validate_checkpoint_against_manifest_rejects_tampered_manifest_hash(tmp_path):
    checkpoint_bytes = b"checkpoint"
    checkpoint = tmp_path / "decoder.pt"
    checkpoint.write_bytes(checkpoint_bytes)
    manifest_path = _manifest_for(checkpoint_bytes).save(tmp_path / "manifest.json")
    payload = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        payload.replace('"model_version": "0.1.0"', '"model_version": "0.2.0"'),
        encoding="utf-8",
    )

    with pytest.raises(ModelManifestError, match="manifest hash mismatch"):
        validate_checkpoint_against_native_model_manifest(checkpoint, manifest_path)


def test_validate_checkpoint_against_manifest_rejects_newer_component_contract(tmp_path):
    checkpoint_bytes = b"checkpoint"
    checkpoint = tmp_path / "decoder.pt"
    checkpoint.write_bytes(checkpoint_bytes)
    manifest_path = _manifest_for(checkpoint_bytes, contract_version=2).save(
        tmp_path / "manifest.json"
    )

    with pytest.raises(ModelManifestError, match="refusing incompatible"):
        validate_checkpoint_against_native_model_manifest(checkpoint, manifest_path)


def test_validate_checkpoint_against_manifest_requires_named_component(tmp_path):
    checkpoint_bytes = b"checkpoint"
    checkpoint = tmp_path / "decoder.pt"
    checkpoint.write_bytes(checkpoint_bytes)
    manifest = NativeModelManifest(
        model_id="cineos-native-video-test",
        model_version="0.1.0",
        runtime_contract_version=1,
        components=(
            NativeModelComponent(
                name="temporal_model",
                version="0.1.0",
                artifact_sha256=hashlib.sha256(b"temporal").hexdigest(),
            ),
        ),
    )
    manifest_path = manifest.save(tmp_path / "manifest.json")

    with pytest.raises(ModelManifestError, match="unsupported component"):
        validate_checkpoint_against_native_model_manifest(checkpoint, manifest_path)
