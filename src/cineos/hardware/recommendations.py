"""Conservative renderer recommendation policy."""

from __future__ import annotations

from .models import GPUInfo, Recommendation

GIB = 1024**3


def recommend(gpus: tuple[GPUInfo, ...], total_ram_bytes: int | None) -> Recommendation:
    """Select guidance from known GPU memory; never assume unknown capacity."""

    capable = tuple(gpu for gpu in gpus if (gpu.vram_bytes or 0) >= 4 * GIB)
    maximum = max((gpu.vram_bytes or 0 for gpu in gpus), default=0)
    prefix = "Guidance only; verify renderer and model requirements before use. "
    if len(capable) >= 2:
        return Recommendation(
            "multi-gpu",
            "local-multi-gpu",
            prefix + "Multiple GPUs have at least 4 GiB VRAM.",
        )
    if maximum >= 16 * GIB:
        return Recommendation(
            "high-memory",
            "local-high-memory",
            prefix + "At least 16 GiB VRAM was detected.",
        )
    if maximum >= 8 * GIB:
        return Recommendation(
            "standard-local",
            "local-standard",
            prefix + "At least 8 GiB VRAM was detected.",
        )
    if maximum >= 4 * GIB:
        return Recommendation(
            "low-memory",
            "local-low-memory",
            prefix + "Use reduced resolution and memory-saving settings.",
        )
    if total_ram_bytes is not None and total_ram_bytes < 4 * GIB:
        return Recommendation(
            "unsupported",
            "none",
            prefix
            + "Available hardware is below conservative local rendering thresholds.",
        )
    return Recommendation(
        "preview-only",
        "preview",
        prefix + "No GPU with at least 4 GiB known VRAM was detected; "
        "use the CPU preview renderer.",
    )
