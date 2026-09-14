"""Fail-closed deployment for fully learned, manifest-bound native temporal video.

Production deployment must bind both the RGB decoder and the temporal generator to
one versioned native-model manifest.  This prevents a release from combining a
trained decoder with the deterministic bootstrap temporal model while still
appearing production-ready.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from cineos.native_image.model_manifest import (
    ModelManifestError,
    NativeModelManifest,
    check_runtime_compatibility,
)

from .deployment import (
    DEFAULT_DECODER_COMPONENT_NAME,
    NATIVE_VIDEO_RUNTIME_CONTRACT_VERSION,
    build_checkpoint_temporal_shot_renderer,
)
from .runtime import NativeTemporalRuntime
from .temporal_model_checkpoint import TemporalModelCheckpoint

DEFAULT_TEMPORAL_COMPONENT_NAME = "temporal_model"
DEFAULT_FULLY_LEARNED_COMPONENT_CONTRACTS = {
    DEFAULT_DECODER_COMPONENT_NAME: 1,
    DEFAULT_TEMPORAL_COMPONENT_NAME: 1,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_manifest_component(
    manifest: NativeModelManifest,
    *,
    component_name: str,
    artifact_path: Path,
) -> None:
    component = next(
        (item for item in manifest.components if item.name == component_name),
        None,
    )
    if component is None:
        raise ModelManifestError(
            f"native model manifest has no required component: {component_name}"
        )
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"native model component does not exist: {artifact_path}"
        )
    if artifact_path.stat().st_size <= 0:
        raise ModelManifestError(
            f"native model component must not be empty: {component_name}"
        )
    if _sha256_file(artifact_path) != component.artifact_sha256:
        raise ModelManifestError(
            f"artifact digest does not match native model component {component_name}"
        )


def validate_fully_learned_native_video_release(
    decoder_checkpoint_path: str | Path,
    temporal_checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    decoder_component_name: str = DEFAULT_DECODER_COMPONENT_NAME,
    temporal_component_name: str = DEFAULT_TEMPORAL_COMPONENT_NAME,
    runtime_contract_version: int = NATIVE_VIDEO_RUNTIME_CONTRACT_VERSION,
    supported_component_contracts: dict[str, int] | None = None,
) -> tuple[NativeModelManifest, TemporalModelCheckpoint]:
    """Validate one release containing both learned decoder and temporal weights."""
    if not decoder_component_name.strip() or not temporal_component_name.strip():
        raise ValueError("component names must not be empty")
    if decoder_component_name == temporal_component_name:
        raise ValueError("decoder and temporal components must have distinct names")
    if runtime_contract_version < 1:
        raise ValueError("runtime_contract_version must be >= 1")

    decoder_path = Path(decoder_checkpoint_path)
    temporal_path = Path(temporal_checkpoint_path)
    manifest = NativeModelManifest.load(manifest_path, verify_hash=True)
    supported = (
        dict(DEFAULT_FULLY_LEARNED_COMPONENT_CONTRACTS)
        if supported_component_contracts is None
        else dict(supported_component_contracts)
    )
    compatibility = check_runtime_compatibility(
        manifest,
        runtime_contract_version=runtime_contract_version,
        supported_component_contracts=supported,
    )
    if not compatibility.compatible:
        raise ModelManifestError(
            "refusing incompatible fully learned native video deployment: "
            + compatibility.reason
        )

    _require_manifest_component(
        manifest,
        component_name=decoder_component_name,
        artifact_path=decoder_path,
    )
    _require_manifest_component(
        manifest,
        component_name=temporal_component_name,
        artifact_path=temporal_path,
    )

    temporal_checkpoint = TemporalModelCheckpoint.load(
        temporal_path,
        verify_hash=True,
    )
    return manifest, temporal_checkpoint


def build_fully_manifest_bound_temporal_shot_renderer(
    decoder_checkpoint_path: str | Path,
    temporal_checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    device: str = "cpu",
    fps: int = 8,
    ffmpeg_binary: str = "ffmpeg",
    max_frames: int = 2400,
    decoder_component_name: str = DEFAULT_DECODER_COMPONENT_NAME,
    temporal_component_name: str = DEFAULT_TEMPORAL_COMPONENT_NAME,
    runtime_contract_version: int = NATIVE_VIDEO_RUNTIME_CONTRACT_VERSION,
    supported_component_contracts: dict[str, int] | None = None,
):
    """Build production video only from jointly manifest-bound learned artifacts.

    The temporal checkpoint must carry positive training provenance and pass its own
    content hash before its weights are installed.  The decoder builder then checks
    latent dimensional compatibility against that restored temporal model, so a
    mixed or stale release fails before any film frame is generated.
    """
    manifest, temporal_checkpoint = validate_fully_learned_native_video_release(
        decoder_checkpoint_path,
        temporal_checkpoint_path,
        manifest_path,
        decoder_component_name=decoder_component_name,
        temporal_component_name=temporal_component_name,
        runtime_contract_version=runtime_contract_version,
        supported_component_contracts=supported_component_contracts,
    )
    runtime = NativeTemporalRuntime.default(model=temporal_checkpoint.model)
    renderer = build_checkpoint_temporal_shot_renderer(
        decoder_checkpoint_path,
        runtime=runtime,
        device=device,
        fps=fps,
        ffmpeg_binary=ffmpeg_binary,
        max_frames=max_frames,
    )
    return renderer, manifest, temporal_checkpoint


__all__ = [
    "DEFAULT_FULLY_LEARNED_COMPONENT_CONTRACTS",
    "DEFAULT_TEMPORAL_COMPONENT_NAME",
    "build_fully_manifest_bound_temporal_shot_renderer",
    "validate_fully_learned_native_video_release",
]
