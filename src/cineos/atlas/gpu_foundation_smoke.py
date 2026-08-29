"""Fail-closed GPU execution for one real pretrained-foundation video shot.

This module binds CINEOS GPU preflight, an explicitly external pretrained
foundation profile, Diffusers execution, and output-artifact evidence into one
operation. It is intentionally small: success means a renderer actually wrote a
fresh, non-empty video artifact for the current request. Planning, model
construction, or a stale artifact from an earlier run is never reported as a
successful GPU render.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .diffusers_video import DiffusersVideoResult
from .foundation_profiles import FoundationExecutionProfile
from .gpu_preflight import (
    GPUExecutionPlan,
    inspect_cuda_environment,
    select_gpu_execution,
)
from .native_request import NativeShotRequest


class GPUFoundationExecutionError(RuntimeError):
    """Raised when a planned foundation render fails to produce usable evidence."""


@dataclass(frozen=True, slots=True)
class GPUFoundationExecutionReceipt:
    """Evidence captured after one successful real foundation-backed GPU shot."""

    result: DiffusersVideoResult
    execution_plan: GPUExecutionPlan
    profile_id: str
    origin: str
    output_bytes: int
    output_sha256: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.result.shot_id,
            "scene_id": self.result.scene_id,
            "request_hash": self.result.request_hash,
            "output_path": self.result.output_path,
            "output_bytes": self.output_bytes,
            "output_sha256": self.output_sha256,
            "frame_count": self.result.frame_count,
            "seed": self.result.seed,
            "elapsed_seconds": self.elapsed_seconds,
            "profile_id": self.profile_id,
            "origin": self.origin,
            "foundation": self.result.foundation.to_dict(),
            "execution_plan": {
                "device": self.execution_plan.device,
                "dtype": self.execution_plan.dtype,
                "memory_strategy": self.execution_plan.memory_strategy,
                "enable_vae_tiling": self.execution_plan.enable_vae_tiling,
                "enable_vae_slicing": self.execution_plan.enable_vae_slicing,
                "enable_attention_slicing": self.execution_plan.enable_attention_slicing,
                "estimated_model_vram_gb": self.execution_plan.estimated_model_vram_gb,
                "observed_total_vram_gb": self.execution_plan.observed_total_vram_gb,
                "observed_free_vram_gb": self.execution_plan.observed_free_vram_gb,
                "fit_margin_gb": self.execution_plan.fit_margin_gb,
            },
        }


def _expected_artifact_path(
    request: NativeShotRequest, output_dir: str | Path
) -> Path:
    return Path(output_dir) / f"{request.scene_id}-{request.shot_id}.mp4"


def _remove_stale_expected_artifact(path: Path) -> None:
    """Ensure an earlier render can never satisfy evidence for the current run."""
    if not path.exists():
        return
    if not path.is_file():
        raise GPUFoundationExecutionError(
            f"expected GPU output path is not a file: {path}"
        )
    try:
        path.unlink()
    except OSError as exc:
        raise GPUFoundationExecutionError(
            f"cannot remove stale GPU output artifact before render: {path}"
        ) from exc


def _validate_result_identity(
    request: NativeShotRequest,
    profile: FoundationExecutionProfile,
    result: DiffusersVideoResult,
    expected_artifact: Path,
) -> Path:
    """Bind returned evidence to the exact request/profile executed this run."""
    if result.shot_id != request.shot_id or result.scene_id != request.scene_id:
        raise GPUFoundationExecutionError(
            "renderer result identity does not match the requested scene/shot"
        )
    if result.request_hash != request.content_hash:
        raise GPUFoundationExecutionError(
            "renderer result request hash does not match the current request"
        )
    if result.seed != request.deterministic_seed:
        raise GPUFoundationExecutionError(
            "renderer result seed does not match the current request"
        )
    if result.foundation != profile.provenance:
        raise GPUFoundationExecutionError(
            "renderer result foundation provenance does not match the selected profile"
        )
    if result.frame_count <= 0:
        raise GPUFoundationExecutionError("renderer reported no generated video frames")

    artifact = Path(result.output_path)
    try:
        actual_path = artifact.resolve(strict=False)
        expected_path = expected_artifact.resolve(strict=False)
    except OSError as exc:
        raise GPUFoundationExecutionError(
            f"cannot resolve renderer output path: {artifact}"
        ) from exc
    if actual_path != expected_path:
        raise GPUFoundationExecutionError(
            "renderer output path does not match the current shot artifact contract"
        )
    return artifact


def execute_foundation_gpu_shot(
    request: NativeShotRequest,
    profile: FoundationExecutionProfile,
    *,
    output_dir: str | Path,
    estimated_model_vram_gb: float | None = None,
    prefer_bfloat16: bool = True,
    torch_module: Any | None = None,
    reference_loader: Any | None = None,
    pipeline_factory: Any | None = None,
    video_exporter: Any | None = None,
) -> GPUFoundationExecutionReceipt:
    """Render one shot only after a real CUDA preflight selects a safe device.

    The function accepts injectable runtime boundaries for regression tests, but
    production callers normally leave them unset so torch, Diffusers and the video
    exporter are loaded from the installed ``video`` extra.
    """
    estimated_vram = (
        profile.minimum_gpu_vram_gb
        if estimated_model_vram_gb is None
        else float(estimated_model_vram_gb)
    )
    devices = inspect_cuda_environment(torch_module=torch_module)
    plan = select_gpu_execution(
        devices,
        estimated_model_vram_gb=estimated_vram,
        prefer_bfloat16=prefer_bfloat16,
    )
    renderer = profile.renderer(
        output_dir=output_dir,
        reference_loader=reference_loader,
        pipeline_factory=pipeline_factory,
        video_exporter=video_exporter,
    )

    expected_artifact = _expected_artifact_path(request, output_dir)
    _remove_stale_expected_artifact(expected_artifact)

    started = perf_counter()
    renderer.initialize()
    try:
        renderer.load_model(**plan.renderer_options())
        renderer.warmup()
        result = renderer.render(request)
    finally:
        renderer.shutdown()
    elapsed = perf_counter() - started

    artifact = _validate_result_identity(
        request,
        profile,
        result,
        expected_artifact,
    )
    try:
        output_bytes = artifact.stat().st_size
    except OSError as exc:
        raise GPUFoundationExecutionError(
            f"renderer reported {artifact} but no readable video artifact exists"
        ) from exc
    if output_bytes <= 0:
        raise GPUFoundationExecutionError(
            f"renderer produced an empty video artifact at {artifact}"
        )

    digest = hashlib.sha256()
    try:
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GPUFoundationExecutionError(
            f"rendered video artifact cannot be hashed: {artifact}"
        ) from exc

    return GPUFoundationExecutionReceipt(
        result=result,
        execution_plan=plan,
        profile_id=profile.profile_id,
        origin=profile.origin,
        output_bytes=output_bytes,
        output_sha256=digest.hexdigest(),
        elapsed_seconds=elapsed,
    )


__all__ = [
    "GPUFoundationExecutionError",
    "GPUFoundationExecutionReceipt",
    "execute_foundation_gpu_shot",
]
