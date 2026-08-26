"""Fail-closed deployment helpers for trained native CINEOS video components."""

from __future__ import annotations

from pathlib import Path

from .learned_decoder import CheckpointLatentRGBDecoder
from .runtime import NativeTemporalRuntime
from .shot_renderer import CINEOSNativeTemporalShotRenderer


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


__all__ = ["build_checkpoint_temporal_shot_renderer"]
