"""GPU preflight and execution planning for real native video inference.

The planner is intentionally model-agnostic.  It does not pretend that a GPU is
available, and it never turns an estimated memory fit into proof of successful
inference.  Its job is to make the execution boundary explicit before CINEOS
loads a large pretrained video foundation.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

_GIB = 1024**3


class GPUPreflightError(RuntimeError):
    """Raised when the requested native GPU execution cannot be planned safely."""


@dataclass(frozen=True, slots=True)
class GPUDeviceProfile:
    """Observed CUDA device properties relevant to video generation."""

    index: int
    name: str
    compute_capability: tuple[int, int]
    total_vram_gb: float
    free_vram_gb: float | None
    supports_bfloat16: bool


@dataclass(frozen=True, slots=True)
class GPUExecutionPlan:
    """Conservative runtime policy for a declared model memory requirement."""

    device: str
    dtype: str
    memory_strategy: str
    enable_vae_tiling: bool
    enable_vae_slicing: bool
    enable_attention_slicing: bool
    estimated_model_vram_gb: float
    observed_total_vram_gb: float
    observed_free_vram_gb: float | None
    fit_margin_gb: float

    def renderer_options(self) -> dict[str, object]:
        """Return options accepted by :class:`DiffusersVideoRenderer`."""
        return {
            "device": self.device,
            "dtype": self.dtype,
            "memory_strategy": self.memory_strategy,
            "enable_vae_tiling": self.enable_vae_tiling,
            "enable_vae_slicing": self.enable_vae_slicing,
            "enable_attention_slicing": self.enable_attention_slicing,
        }


def inspect_cuda_environment(
    torch_module: Any | None = None,
) -> tuple[GPUDeviceProfile, ...]:
    """Inspect real CUDA devices without importing torch at package import time."""
    torch = torch_module
    if torch is None:
        try:
            torch = import_module("torch")
        except ImportError as exc:
            raise GPUPreflightError(
                "GPU preflight requires the optional 'video' dependencies"
            ) from exc

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise GPUPreflightError("torch reports that CUDA is unavailable")

    device_count = int(cuda.device_count())
    if device_count < 1:
        raise GPUPreflightError("CUDA is available but no CUDA devices were found")

    profiles: list[GPUDeviceProfile] = []
    bf16_supported = _supports_bfloat16(cuda)
    current_device = _current_device(cuda)
    for index in range(device_count):
        properties = cuda.get_device_properties(index)
        capability = cuda.get_device_capability(index)
        free_vram = _free_vram_gb(cuda, index, current_device)
        profiles.append(
            GPUDeviceProfile(
                index=index,
                name=str(properties.name),
                compute_capability=(int(capability[0]), int(capability[1])),
                total_vram_gb=float(properties.total_memory) / _GIB,
                free_vram_gb=free_vram,
                supports_bfloat16=bf16_supported,
            )
        )
    return tuple(profiles)


def plan_gpu_execution(
    profile: GPUDeviceProfile,
    *,
    estimated_model_vram_gb: float,
    prefer_bfloat16: bool = True,
) -> GPUExecutionPlan:
    """Choose a conservative Diffusers memory policy for one observed GPU.

    ``estimated_model_vram_gb`` is an operator/model-card estimate, not a value
    invented by CINEOS.  The thresholds deliberately leave headroom for activations,
    references and video latents.  When live free-VRAM telemetry is available it is
    used instead of total capacity so a busy GPU cannot be incorrectly approved for
    a render that is already certain to overcommit memory.
    """
    if estimated_model_vram_gb <= 0:
        raise ValueError("estimated_model_vram_gb must be positive")
    if profile.total_vram_gb <= 0:
        raise GPUPreflightError("GPU reports no usable VRAM")
    if profile.free_vram_gb is not None:
        if profile.free_vram_gb <= 0:
            raise GPUPreflightError("GPU reports no free VRAM")
        if profile.free_vram_gb > profile.total_vram_gb:
            raise GPUPreflightError("GPU free VRAM exceeds reported total VRAM")

    usable_vram_gb = (
        profile.free_vram_gb
        if profile.free_vram_gb is not None
        else profile.total_vram_gb
    )
    ratio = usable_vram_gb / estimated_model_vram_gb
    if ratio >= 1.20:
        strategy = "resident"
        tiling = False
        slicing = False
        attention_slicing = False
    elif ratio >= 0.55:
        strategy = "model_cpu_offload"
        tiling = True
        slicing = True
        attention_slicing = False
    elif ratio >= 0.30:
        strategy = "sequential_cpu_offload"
        tiling = True
        slicing = True
        attention_slicing = True
    else:
        raise GPUPreflightError(
            "observed free GPU VRAM is below the conservative minimum for this model; "
            "free GPU memory, use a smaller foundation, or use a larger GPU"
        )

    dtype = "bfloat16" if prefer_bfloat16 and profile.supports_bfloat16 else "float16"
    return GPUExecutionPlan(
        device=f"cuda:{profile.index}",
        dtype=dtype,
        memory_strategy=strategy,
        enable_vae_tiling=tiling,
        enable_vae_slicing=slicing,
        enable_attention_slicing=attention_slicing,
        estimated_model_vram_gb=float(estimated_model_vram_gb),
        observed_total_vram_gb=profile.total_vram_gb,
        observed_free_vram_gb=profile.free_vram_gb,
        fit_margin_gb=usable_vram_gb - float(estimated_model_vram_gb),
    )


def select_gpu_execution(
    profiles: tuple[GPUDeviceProfile, ...],
    *,
    estimated_model_vram_gb: float,
    prefer_bfloat16: bool = True,
) -> GPUExecutionPlan:
    """Select the safest usable GPU instead of assuming device zero.

    Candidate plans are ranked first by memory strategy (resident is preferred over
    model offload, which is preferred over sequential offload), then by measured free
    memory headroom.  Devices that cannot satisfy the conservative working-set floor
    are skipped.  This keeps multi-GPU render workers from choosing a busy card when a
    safer device is available.
    """
    if not profiles:
        raise GPUPreflightError("no CUDA device profiles were supplied")

    strategy_rank = {
        "resident": 3,
        "model_cpu_offload": 2,
        "sequential_cpu_offload": 1,
    }
    candidates: list[GPUExecutionPlan] = []
    rejected: list[str] = []
    for profile in profiles:
        try:
            candidates.append(
                plan_gpu_execution(
                    profile,
                    estimated_model_vram_gb=estimated_model_vram_gb,
                    prefer_bfloat16=prefer_bfloat16,
                )
            )
        except GPUPreflightError as exc:
            rejected.append(f"cuda:{profile.index}: {exc}")

    if not candidates:
        detail = "; ".join(rejected) if rejected else "no usable candidates"
        raise GPUPreflightError(f"no GPU can safely execute the model: {detail}")

    def score(plan: GPUExecutionPlan) -> tuple[int, int, float, float]:
        telemetry_rank = 1 if plan.observed_free_vram_gb is not None else 0
        free_or_total = (
            plan.observed_free_vram_gb
            if plan.observed_free_vram_gb is not None
            else plan.observed_total_vram_gb
        )
        return (
            strategy_rank[plan.memory_strategy],
            telemetry_rank,
            plan.fit_margin_gb,
            free_or_total,
        )

    return max(candidates, key=score)


def _supports_bfloat16(cuda: Any) -> bool:
    checker = getattr(cuda, "is_bf16_supported", None)
    return bool(checker()) if callable(checker) else False


def _current_device(cuda: Any) -> int | None:
    getter = getattr(cuda, "current_device", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except (RuntimeError, ValueError):
        return None


def _free_vram_gb(cuda: Any, index: int, current_device: int | None) -> float | None:
    mem_get_info = getattr(cuda, "mem_get_info", None)
    if not callable(mem_get_info):
        return None

    try:
        free_bytes, _total_bytes = mem_get_info(index)
        return float(free_bytes) / _GIB
    except TypeError:
        # Older torch/test doubles expose mem_get_info() without a device argument.
        if current_device is None or index != current_device:
            return None
    except (RuntimeError, ValueError):
        return None

    try:
        free_bytes, _total_bytes = mem_get_info()
    except (RuntimeError, TypeError, ValueError):
        return None
    return float(free_bytes) / _GIB
