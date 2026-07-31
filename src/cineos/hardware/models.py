"""Immutable values returned by the hardware diagnostic subsystem."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

RecommendationCategory = Literal[
    "unsupported",
    "preview-only",
    "low-memory",
    "standard-local",
    "high-memory",
    "multi-gpu",
]


@dataclass(frozen=True, slots=True)
class GPUInfo:
    """One graphics adapter, with memory expressed in bytes when discoverable."""

    vendor: str
    model: str
    vram_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Conservative, non-binding renderer selection guidance."""

    category: RecommendationCategory
    renderer: str
    guidance: str


@dataclass(frozen=True, slots=True)
class HardwareReport:
    """A point-in-time description of renderer-relevant local hardware."""

    os: str
    os_version: str
    architecture: str
    python_version: str
    cpu_model: str
    logical_cpu_count: int | None
    physical_cpu_count: int | None
    total_ram_bytes: int | None
    available_ram_bytes: int | None
    gpus: tuple[GPUInfo, ...]
    nvidia_driver_version: str | None
    cuda_version: str | None
    pytorch_installed: bool
    pytorch_cuda_available: bool | None
    ffmpeg_available: bool
    ffmpeg_version: str | None
    free_disk_bytes: int | None
    recommendation: Recommendation

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready mapping with an explicit aggregate GPU count."""

        value = asdict(self)
        value["gpu_count"] = self.gpu_count
        return value
