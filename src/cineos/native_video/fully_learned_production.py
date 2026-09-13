"""Strict composition root for fully learned native CINEOS production films.

This module closes the model-identity gap between shot rendering and long-range
continuity.  The renderer and continuity bridge are constructed from the *same*
manifest-bound learned temporal checkpoint, and the artifact manifest must match the
registry's currently active release before a production FIRST FILM runtime exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cineos.native_image.model_manifest import (
    ModelManifestError,
    NativeModelRegistry,
    check_runtime_compatibility,
)

from .film_bridge import NativeFilmContinuityBridge, temporal_model_fingerprint
from .production_first_film import (
    ProductionFirstFilmRuntime,
    build_released_production_first_film_runtime,
)
from .temporal_deployment import build_fully_manifest_bound_temporal_shot_renderer


def build_fully_learned_production_first_film_runtime(
    decoder_checkpoint_path: str | Path,
    temporal_checkpoint_path: str | Path,
    manifest_path: str | Path,
    model_registry: NativeModelRegistry,
    validator: Any | None = None,
    *,
    device: str = "cpu",
    fps: int = 8,
    ffmpeg_binary: str = "ffmpeg",
    max_frames: int = 2400,
    renderer_id: str = "cineos-native-learned",
    max_recovery_attempts: int = 2,
    final_gate: Any | None = None,
) -> ProductionFirstFilmRuntime:
    """Compose production film generation from one fully verified learned release.

    This is the preferred V1 production entrypoint. It rejects:

    * decoder-only manifests,
    * untrained/bootstrap temporal checkpoints,
    * artifact digest drift,
    * runtime/component contract incompatibility,
    * a manifest that is not the registry's active release, and
    * accidental divergence between renderer temporal weights and continuity memory.
    """
    if not isinstance(model_registry, NativeModelRegistry):
        raise TypeError("model_registry must be a NativeModelRegistry")
    if not device.strip():
        raise ValueError("device must not be empty")

    active = model_registry.active()
    if active is None:
        raise ModelManifestError(
            "fully learned production requires an active native model release"
        )
    compatibility = check_runtime_compatibility(
        active,
        runtime_contract_version=model_registry.runtime_contract_version,
        supported_component_contracts=model_registry.supported_component_contracts,
    )
    if not compatibility.compatible:
        raise ModelManifestError(
            "refusing incompatible active native model release: " + compatibility.reason
        )

    renderer, artifact_manifest, temporal_checkpoint = (
        build_fully_manifest_bound_temporal_shot_renderer(
            decoder_checkpoint_path,
            temporal_checkpoint_path,
            manifest_path,
            device=device,
            fps=fps,
            ffmpeg_binary=ffmpeg_binary,
            max_frames=max_frames,
            runtime_contract_version=model_registry.runtime_contract_version,
            supported_component_contracts=model_registry.supported_component_contracts,
        )
    )
    if artifact_manifest.manifest_sha256 != active.manifest_sha256:
        raise ModelManifestError(
            "artifact manifest does not match active native model registry release"
        )

    continuity = NativeFilmContinuityBridge(
        model=temporal_checkpoint.model,
        device=device,
    )
    renderer_model = renderer.runtime.model
    if temporal_model_fingerprint(renderer_model) != temporal_model_fingerprint(
        continuity.model
    ):
        raise ModelManifestError(
            "renderer and continuity temporal model fingerprints do not match"
        )

    kwargs: dict[str, Any] = {
        "continuity": continuity,
        "renderer_id": renderer_id,
        "max_recovery_attempts": max_recovery_attempts,
        "device": device,
    }
    if final_gate is not None:
        kwargs["final_gate"] = final_gate

    return build_released_production_first_film_runtime(
        renderer,
        model_registry,
        validator,
        **kwargs,
    )


__all__ = ["build_fully_learned_production_first_film_runtime"]
