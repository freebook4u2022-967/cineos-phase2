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

PRODUCTION_EXECUTION_EVIDENCE_SCHEMA = "cineos-wan22-production-execution/1.1"
_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_cuda_device(device: str) -> None:
    normalized = device.strip().lower()
    if not normalized or not normalized.startswith("cuda"):
        raise Wan22ExecutionError(
            "production Wan2.2 validation requires a CUDA device; "
            f"received {device!r}"
        )


def _require_hex_digest(value: Any, *, length: int, label: str) -> str:
    if not isinstance(value, str):
        raise Wan22ExecutionError(f"production execution receipt is missing {label}")
    normalized = value.lower()
    if len(normalized) != length or any(
        character not in _HEX_DIGITS for character in normalized
    ):
        raise Wan22ExecutionError(
            f"production execution receipt contains an invalid {label}"
        )
    return normalized


def _validate_foundation_receipt(receipt: dict[str, Any], *, device: str) -> None:
    if receipt.get("status") != "rendered":
        raise Wan22ExecutionError(
            "production execution did not report a rendered artifact"
        )

    runtime = receipt.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("device") != device:
        raise Wan22ExecutionError(
            "production execution receipt does not bind the requested CUDA device"
        )

    foundation_profile = receipt.get("foundation_profile")
    if not isinstance(foundation_profile, dict):
        raise Wan22ExecutionError(
            "production execution receipt is missing foundation profile"
        )

    expected_profile = WAN22_TI2V_5B_PROFILE.snapshot()
    if foundation_profile.get("origin") != expected_profile["origin"]:
        raise Wan22ExecutionError(
            "production execution receipt foundation origin does not match the pinned profile"
        )
    if foundation_profile.get("profile_id") != expected_profile["profile_id"]:
        raise Wan22ExecutionError(
            "production execution receipt foundation profile id does not match the pinned profile"
        )

    provenance = foundation_profile.get("provenance")
    expected_provenance = expected_profile["provenance"]
    if not isinstance(provenance, dict):
        raise Wan22ExecutionError(
            "production execution receipt is missing foundation provenance"
        )
    for field in ("model_id", "revision", "license_id", "source_url"):
        if provenance.get(field) != expected_provenance.get(field):
            raise Wan22ExecutionError(
                "production execution receipt foundation provenance does not match "
                f"the pinned profile: {field}"
            )

    artifact = receipt.get("artifact")
    if not isinstance(artifact, dict):
        raise Wan22ExecutionError(
            "production execution receipt is missing artifact evidence"
        )
    _require_hex_digest(
        artifact.get("sha256"), length=64, label="SHA-256 artifact binding"
    )
    _require_hex_digest(
        receipt.get("request_hash"), length=64, label="request hash binding"
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
        "foundation_model_id": WAN22_TI2V_5B_PROFILE.provenance.model_id,
        "foundation_revision": WAN22_TI2V_5B_PROFILE.provenance.revision,
        "foundation_license_id": WAN22_TI2V_5B_PROFILE.provenance.license_id,
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
