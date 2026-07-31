"""Top-level hardware collection orchestration."""

from __future__ import annotations

from pathlib import Path

from .models import HardwareReport
from .nvidia import probe_nvidia
from .recommendations import recommend
from .system import probe_other_gpus, probe_system


def _torch_cuda() -> tuple[bool, bool | None]:
    try:
        import torch  # type: ignore[import-not-found]
    except (ImportError, OSError):
        return False, None
    try:
        return True, bool(torch.cuda.is_available())
    except (AttributeError, RuntimeError):
        return True, False


def probe(path: Path | None = None) -> HardwareReport:
    """Return a complete report without requiring optional hardware or packages."""

    system = probe_system(path)
    nvidia = probe_nvidia()
    gpus = nvidia.gpus + probe_other_gpus()
    torch_installed, torch_cuda = _torch_cuda()
    return HardwareReport(
        os=system.os,
        os_version=system.os_version,
        architecture=system.architecture,
        python_version=system.python_version,
        cpu_model=system.cpu_model,
        logical_cpu_count=system.logical_cpu_count,
        physical_cpu_count=system.physical_cpu_count,
        total_ram_bytes=system.total_ram_bytes,
        available_ram_bytes=system.available_ram_bytes,
        gpus=gpus,
        nvidia_driver_version=nvidia.driver_version,
        cuda_version=nvidia.cuda_version,
        pytorch_installed=torch_installed,
        pytorch_cuda_available=torch_cuda,
        ffmpeg_available=system.ffmpeg_version is not None,
        ffmpeg_version=system.ffmpeg_version,
        free_disk_bytes=system.free_disk_bytes,
        recommendation=recommend(gpus, system.total_ram_bytes),
    )
