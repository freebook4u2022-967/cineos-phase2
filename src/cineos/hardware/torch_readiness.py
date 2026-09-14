"""Fail-closed readiness checks for CINEOS PyTorch model deployment.

This module reports whether a requested neural runtime is genuinely usable before
large video weights are allocated.  It intentionally distinguishes software
readiness from real benchmark evidence: a successful probe means the selected
accelerator satisfies declared execution prerequisites, not that film quality or
throughput has been validated on that machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from .torch_device import resolve_torch_device


@dataclass(frozen=True, slots=True)
class TorchDeviceReadiness:
    """Structured preflight result for one neural execution target."""

    requested_device: str
    device: str | None
    dtype: str
    accelerator: str | None
    device_index: int | None
    device_name: str | None
    total_memory_bytes: int | None
    bf16_supported: bool | None
    ready: bool
    failures: tuple[str, ...]

    def require_ready(self) -> None:
        """Raise with stable reason codes when deployment prerequisites fail."""

        if self.ready:
            return
        reasons = ", ".join(self.failures) or "unknown-readiness-failure"
        raise RuntimeError(f"neural device is not deployment-ready: {reasons}")


def probe_torch_device(
    requested_device: str = "auto",
    *,
    dtype: str = "bfloat16",
    minimum_cuda_vram_bytes: int | None = None,
    torch_module: Any | None = None,
) -> TorchDeviceReadiness:
    """Inspect a PyTorch target and return actionable, machine-readable failures.

    ``minimum_cuda_vram_bytes`` is deliberately caller-supplied because different
    video foundations have very different memory footprints.  The preflight does
    not invent a universal model requirement and does not allocate model weights.
    """

    if minimum_cuda_vram_bytes is not None and minimum_cuda_vram_bytes < 0:
        raise ValueError("minimum_cuda_vram_bytes must be non-negative")

    try:
        torch = torch_module or import_module("torch")
    except (ImportError, OSError):
        return _failure(requested_device, dtype, "pytorch-unavailable")

    failures: list[str] = []
    try:
        device = resolve_torch_device(requested_device, torch_module=torch)
    except (RuntimeError, ValueError):
        return _failure(requested_device, dtype, "requested-device-unavailable")

    if getattr(torch, dtype, None) is None:
        failures.append("requested-dtype-unavailable")

    accelerator = device.split(":", 1)[0]
    device_index: int | None = None
    device_name: str | None = None
    total_memory_bytes: int | None = None
    bf16_supported: bool | None = None

    if accelerator == "cuda":
        device_index = _cuda_index(torch, device)
        device_name, total_memory_bytes = _cuda_properties(torch, device_index)
        bf16_supported = _cuda_bf16_supported(torch)
        if dtype == "bfloat16" and bf16_supported is False:
            failures.append("cuda-bfloat16-unsupported")
        if minimum_cuda_vram_bytes is not None:
            if total_memory_bytes is None:
                failures.append("cuda-vram-unknown")
            elif total_memory_bytes < minimum_cuda_vram_bytes:
                failures.append("cuda-vram-below-minimum")

    return TorchDeviceReadiness(
        requested_device=requested_device,
        device=device,
        dtype=dtype,
        accelerator=accelerator,
        device_index=device_index,
        device_name=device_name,
        total_memory_bytes=total_memory_bytes,
        bf16_supported=bf16_supported,
        ready=not failures,
        failures=tuple(failures),
    )


def _failure(requested_device: str, dtype: str, reason: str) -> TorchDeviceReadiness:
    return TorchDeviceReadiness(
        requested_device=requested_device,
        device=None,
        dtype=dtype,
        accelerator=None,
        device_index=None,
        device_name=None,
        total_memory_bytes=None,
        bf16_supported=None,
        ready=False,
        failures=(reason,),
    )


def _cuda_index(torch: Any, device: str) -> int:
    if ":" in device:
        return int(device.split(":", 1)[1])
    try:
        return int(torch.cuda.current_device())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0


def _cuda_properties(torch: Any, index: int) -> tuple[str | None, int | None]:
    try:
        properties = torch.cuda.get_device_properties(index)
    except (AttributeError, RuntimeError):
        return None, None
    name = getattr(properties, "name", None)
    memory = getattr(properties, "total_memory", None)
    try:
        total_memory = int(memory) if memory is not None else None
    except (TypeError, ValueError):
        total_memory = None
    return str(name) if name is not None else None, total_memory


def _cuda_bf16_supported(torch: Any) -> bool | None:
    try:
        check = torch.cuda.is_bf16_supported
    except AttributeError:
        return None
    try:
        return bool(check())
    except (RuntimeError, TypeError):
        return None
