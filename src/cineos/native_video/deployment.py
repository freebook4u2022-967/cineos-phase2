"""Fail-closed deployment helpers for trained native CINEOS video components."""

from __future__ import annotations

import hashlib
from pathlib import Path

from cineos.native_image.model_manifest import (
    ModelManifestError,
    NativeModelManifest,
    check_runtime_compatibility,
)

from .learned_decoder import CheckpointLatentRGBDecoder
from .runtime import NativeTemporalRuntime
from .shot_renderer import CINEOSNativeTemporalShotRenderer

NATIVE_VIDEO_RUNTIME_CONTRACT_VERSION = 1
DEFAULT_DECODER_COMPONENT_NAME = "rgb_decoder"
DEFAULT_SUPPORTED_COMPONENT_CONTRACTS = {DEFAULT_DECODER_COMPONENT_NAME: 1}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint_against_native_model_manifest(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    component_name: str = DEFAULT_DECODER_COMPONENT_NAME,
    runtime_contract_version: int = NATIVE_VIDEO_RUNTIME_CONTRACT_VERSION,
    supported_component_contracts: dict[str, int] | None = None,
) -> NativeModelManifest:
    """Verify that a trained checkpoint is exactly the artifact released by a manifest.

    Production renderer deployment must not trust a checkpoint merely because it can
    be deserialized.  This gate verifies the signed-by-hash native model manifest,
    checks runtime/component contract compatibility, requires the named decoder
    component, and hashes the checkpoint bytes before any learned weights are loaded.

    The function is intentionally independent of torch/CUDA so release tooling and CI
    can verify deployment identity even on machines that cannot execute the model.
    """
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"decoder checkpoint does not exist: {checkpoint}")
    if checkpoint.stat().st_size <= 0:
        raise ModelManifestError("decoder checkpoint must not be empty")
    if not component_name.strip():
        raise ValueError("component_name must not be empty")
    if runtime_contract_version < 1:
        raise ValueError("runtime_contract_version must be >= 1")

    manifest = NativeModelManifest.load(manifest_path, verify_hash=True)
    supported = (
        dict(DEFAULT_SUPPORTED_COMPONENT_CONTRACTS)
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
            "refusing incompatible native video model deployment: "
            + compatibility.reason
        )

    component = next(
        (item for item in manifest.components if item.name == component_name),
        None,
    )
    if component is None:
        raise ModelManifestError(
            f"native model manifest has no required component: {component_name}"
        )
    actual_digest = _sha256_file(checkpoint)
    if actual_digest != component.artifact_sha256:
        raise ModelManifestError(
            f"checkpoint digest does not match native model component {component_name}"
        )
    return manifest


def build_checkpoint_temporal_shot_renderer(
    checkpoint_path: str | Path,
    *,
    runtime: NativeTemporalRuntime | None = None,
    device: str = "cpu",
    fps: int = 8,
    ffmpeg_binary: str = "ffmpeg",
    max_frames: int = 2400,
) -> CINEOSNativeTemporalShotRenderer:
    """Compose a native temporal renderer with trained CINEOS RGB weights.

    The checkpoint determines output resolution and decoder latent dimensionality.
    Deployment fails before film work begins when those learned weights are
    incompatible with the active temporal model.  This prevents silent truncation,
    padding or fallback to the analytic bootstrap decoder in production.

    This compatibility helper remains available for development and migration. New
    production entrypoints should use ``build_manifest_bound_temporal_shot_renderer``
    so the checkpoint is also bound to a versioned native-model release manifest.
    """
    active_runtime = runtime or NativeTemporalRuntime.default()
    decoder = CheckpointLatentRGBDecoder(checkpoint_path, device=device)
    temporal_latent_dim = active_runtime.model.latent_dim
    if decoder.latent_dim != temporal_latent_dim:
        raise ValueError(
            "trained RGB decoder is incompatible with temporal model: "
            f"decoder latent_dim={decoder.latent_dim}, "
            f"temporal latent_dim={temporal_latent_dim}"
        )
    return CINEOSNativeTemporalShotRenderer(
        runtime=active_runtime,
        decoder=decoder,
        width=decoder.width,
        height=decoder.height,
        fps=fps,
        ffmpeg_binary=ffmpeg_binary,
        max_frames=max_frames,
    )


def build_manifest_bound_temporal_shot_renderer(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    *,
    runtime: NativeTemporalRuntime | None = None,
    device: str = "cpu",
    fps: int = 8,
    ffmpeg_binary: str = "ffmpeg",
    max_frames: int = 2400,
    component_name: str = DEFAULT_DECODER_COMPONENT_NAME,
    runtime_contract_version: int = NATIVE_VIDEO_RUNTIME_CONTRACT_VERSION,
    supported_component_contracts: dict[str, int] | None = None,
) -> tuple[CINEOSNativeTemporalShotRenderer, NativeModelManifest]:
    """Build a production renderer only from manifest-bound learned weights.

    Returning the verified manifest alongside the renderer gives orchestration code
    the exact ``manifest_sha256`` needed for ``ProductionRuntimeManifest`` and final
    film audit binding.  This closes the deployment provenance chain from released
    model artifact -> runtime identity -> final-film acceptance evidence.
    """
    manifest = validate_checkpoint_against_native_model_manifest(
        checkpoint_path,
        manifest_path,
        component_name=component_name,
        runtime_contract_version=runtime_contract_version,
        supported_component_contracts=supported_component_contracts,
    )
    renderer = build_checkpoint_temporal_shot_renderer(
        checkpoint_path,
        runtime=runtime,
        device=device,
        fps=fps,
        ffmpeg_binary=ffmpeg_binary,
        max_frames=max_frames,
    )
    return renderer, manifest


__all__ = [
    "DEFAULT_DECODER_COMPONENT_NAME",
    "DEFAULT_SUPPORTED_COMPONENT_CONTRACTS",
    "NATIVE_VIDEO_RUNTIME_CONTRACT_VERSION",
    "build_checkpoint_temporal_shot_renderer",
    "build_manifest_bound_temporal_shot_renderer",
    "validate_checkpoint_against_native_model_manifest",
]
