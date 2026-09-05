"""Fail-closed production entry point for real Wan2.2 GPU execution.

Wan2.2 remains an external pretrained foundation.  This module deliberately
separates the production execution boundary from ``wan22_execution``, whose
factory/exporter injection hooks exist for regression testing.  Production
callers get no such injection surface, so a successful receipt cannot silently
turn a test double into evidence of real model execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .foundation_profiles import WAN22_TI2V_5B_PROFILE
from .wan22_execution import (
    Wan22ExecutionConfig,
    Wan22ExecutionError,
    run_wan22_gpu_validation,
)

PRODUCTION_EXECUTION_EVIDENCE_SCHEMA = "cineos-wan22-production-execution/1.0"


def _require_cuda_device(device: str) -> None:
    normalized = device.strip().lower()
    if not normalized or not normalized.startswith("cuda"):
        raise Wan22ExecutionError(
            "production Wan2.2 validation requires a CUDA device; "
            f"received {device!r}"
        )


def _validate_foundation_receipt(receipt: dict[str, Any], *, device: str) -> None:
    if receipt.get("status") != "rendered":
        raise Wan22ExecutionError("production execution did not report a rendered artifact")

    runtime = receipt.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("device") != device:
        raise Wan22ExecutionError(
            "production execution receipt does not bind the requested CUDA device"
        )

    foundation_profile = receipt.get("foundation_profile")
    if not isinstance(foundation_profile, dict):
        raise Wan22ExecutionError("production execution receipt is missing foundation profile")
    if foundation_profile.get("origin") != WAN22_TI2V_5B_PROFILE.origin:
        raise Wan22ExecutionError(
            "production execution receipt foundation origin does not match the pinned profile"
        )

    artifact = receipt.get("artifact")
    if not isinstance(artifact, dict):
        raise Wan22ExecutionError("production execution receipt is missing artifact evidence")
    digest = artifact.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise Wan22ExecutionError(
            "production execution receipt is missing a SHA-256 artifact binding"
        )


def run_wan22_production_validation(
    config: Wan22ExecutionConfig,
    *,
    output_dir: str | Path,
    device: str = "cuda",
    memory_strategy: str = "model_cpu_offload",
    dtype: str = "bfloat16",
    enable_vae_tiling: bool = True,
    reference_loader: Any | None = None,
) -> dict[str, Any]:
    """Run the pinned external Wan2.2 foundation through a production-only gate.

    Unlike the lower-level regression harness, this API intentionally exposes no
    ``pipeline_factory`` or ``video_exporter`` override.  The only successful
    execution path therefore invokes the real Diffusers-backed renderer from the
    pinned foundation profile.  ``reference_loader`` remains injectable because
    approved identity assets are deployment data, not a substitute renderer.
    """

    _require_cuda_device(device)
    receipt = run_wan22_gpu_validation(
        config,
        output_dir=output_dir,
        device=device,
        memory_strategy=memory_strategy,
        dtype=dtype,
        enable_vae_tiling=enable_vae_tiling,
        reference_loader=reference_loader,
    )
    _validate_foundation_receipt(receipt, device=device)

    production_receipt = dict(receipt)
    production_receipt["execution_evidence"] = {
        "schema": PRODUCTION_EXECUTION_EVIDENCE_SCHEMA,
        "classification": "external_pretrained_foundation",
        "foundation_profile_id": WAN22_TI2V_5B_PROFILE.profile_id,
        "foundation_origin": WAN22_TI2V_5B_PROFILE.origin,
        "injected_pipeline_factory": False,
        "injected_video_exporter": False,
        "cuda_required": True,
        "device": device,
    }
    return production_receipt


__all__ = [
    "PRODUCTION_EXECUTION_EVIDENCE_SCHEMA",
    "run_wan22_production_validation",
]
