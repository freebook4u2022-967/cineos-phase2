"""Real Wan2.2 TI2V 5B execution harness for CINEOS GPU validation.

This module is intentionally explicit that Wan2.2 is an external pretrained
foundation. It provides a reproducible path from a CINEOS-native shot request to
real Diffusers execution without relabelling the underlying checkpoint as a
CINEOS-native model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .foundation_profiles import WAN22_TI2V_5B_PROFILE
from .native_request import NativeShotRequest

WAN22_TEMPORAL_COMPRESSION = 4
WAN22_FRAME_REMAINDER = 1


@dataclass(frozen=True, slots=True)
class Wan22ExecutionConfig:
    """Deterministic settings for one real Wan2.2 validation shot."""

    prompt: str
    seed: int = 42
    requested_duration_seconds: float = 5.0
    fps: float = 24.0
    width: int = 1280
    height: int = 704
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    negative_prompt: str = (
        "low quality, blurry, deformed anatomy, extra fingers, duplicate people, "
        "identity drift, temporal flicker, warped geometry"
    )

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.requested_duration_seconds <= 0:
            raise ValueError("requested_duration_seconds must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")


def aligned_wan22_frame_count(duration_seconds: float, fps: float = 24.0) -> int:
    """Round a requested duration up to Wan2.2's 4k+1 temporal frame contract.

    Wan2.2 TI2V uses temporal compression by four. Real-world Wan2.2 execution
    commonly uses frame counts such as 81 or 121, i.e. counts congruent to one
    modulo four. Rounding upward preserves the requested minimum duration and
    avoids silently shortening a shot.
    """

    if duration_seconds <= 0 or fps <= 0:
        raise ValueError("duration_seconds and fps must be positive")
    requested = max(1, math.ceil(duration_seconds * fps))
    remainder = requested % WAN22_TEMPORAL_COMPRESSION
    delta = (WAN22_FRAME_REMAINDER - remainder) % WAN22_TEMPORAL_COMPRESSION
    return requested + delta


def build_wan22_execution_request(config: Wan22ExecutionConfig) -> NativeShotRequest:
    """Build a deterministic CINEOS request aligned for the pinned Wan2.2 profile."""

    frames = aligned_wan22_frame_count(
        config.requested_duration_seconds,
        config.fps,
    )
    execution_duration = frames / config.fps
    request = NativeShotRequest(
        shot_id="wan22-gpu-validation-shot",
        scene_id="wan22-gpu-validation",
        camera={
            "resolution": (config.width, config.height),
            "fps": config.fps,
            "duration": execution_duration,
            "shot_size": "medium-wide",
            "movement": "controlled tracking",
            "lens": "35mm cinematic",
        },
        characters=[],
        environment={
            "name": "GPU validation environment",
            "description": "photorealistic cinematic environment with stable geometry",
        },
        wardrobe=[],
        props=[],
        continuity={
            "mode": "single-shot-foundation-validation",
        },
        performance={},
        approved_reference_ids=[],
        deterministic_seed=config.seed,
        renderer_requirements={
            "inference": {
                "num_inference_steps": config.num_inference_steps,
                "guidance_scale": config.guidance_scale,
            }
        },
        metadata={
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "requested_duration_seconds": config.requested_duration_seconds,
            "execution_duration_seconds": execution_duration,
            "execution_frame_count": frames,
            "foundation_origin": WAN22_TI2V_5B_PROFILE.origin,
            "foundation_profile_id": WAN22_TI2V_5B_PROFILE.profile_id,
        },
    )
    request.refresh_hash()
    return request


def run_wan22_gpu_validation(
    config: Wan22ExecutionConfig,
    *,
    output_dir: str | Path,
    device: str = "cuda",
    memory_strategy: str = "model_cpu_offload",
    dtype: str = "bfloat16",
    enable_vae_tiling: bool = True,
    pipeline_factory: Any | None = None,
    video_exporter: Any | None = None,
) -> dict[str, Any]:
    """Execute one real foundation shot and return an auditable receipt.

    ``pipeline_factory`` and ``video_exporter`` exist for regression tests. When
    omitted, the function loads the pinned external Wan2.2 checkpoint through
    Diffusers and therefore requires the optional video dependencies plus a GPU.
    """

    request = build_wan22_execution_request(config)
    renderer = WAN22_TI2V_5B_PROFILE.renderer(
        output_dir=output_dir,
        pipeline_factory=pipeline_factory,
        video_exporter=video_exporter,
    )
    renderer.initialize()
    try:
        renderer.load_model(
            device=device,
            dtype=dtype,
            memory_strategy=memory_strategy,
            enable_vae_tiling=enable_vae_tiling,
        )
        renderer.warmup()
        result = renderer.render(request)
    finally:
        renderer.shutdown()

    return {
        "status": "rendered",
        "shot_id": result.shot_id,
        "scene_id": result.scene_id,
        "output_path": result.output_path,
        "actual_frame_count": result.frame_count,
        "requested_duration_seconds": config.requested_duration_seconds,
        "execution_duration_seconds": request.metadata["execution_duration_seconds"],
        "execution_frame_count": request.metadata["execution_frame_count"],
        "fps": config.fps,
        "seed": result.seed,
        "request_hash": result.request_hash,
        "foundation_profile": WAN22_TI2V_5B_PROFILE.snapshot(),
        "foundation": result.foundation.to_dict(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real CINEOS -> Wan2.2 Diffusers GPU validation shot."
    )
    parser.add_argument("prompt")
    parser.add_argument("--output-dir", default="artifacts/wan22-gpu-validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--memory-strategy",
        choices=("resident", "model_cpu_offload", "sequential_cpu_offload"),
        default="model_cpu_offload",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = Wan22ExecutionConfig(
        prompt=args.prompt,
        requested_duration_seconds=args.duration,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )
    receipt = run_wan22_gpu_validation(
        config,
        output_dir=args.output_dir,
        device=args.device,
        memory_strategy=args.memory_strategy,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "WAN22_FRAME_REMAINDER",
    "WAN22_TEMPORAL_COMPRESSION",
    "Wan22ExecutionConfig",
    "aligned_wan22_frame_count",
    "build_wan22_execution_request",
    "run_wan22_gpu_validation",
]
