"""Optional NVIDIA command-line adapter."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import GPUInfo
from .system import run_command


@dataclass(frozen=True, slots=True)
class NvidiaInfo:
    gpus: tuple[GPUInfo, ...] = ()
    driver_version: str | None = None
    cuda_version: str | None = None


def probe_nvidia() -> NvidiaInfo:
    """Query ``nvidia-smi`` if present; absence is a normal CPU-only result."""

    rows = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not rows:
        return NvidiaInfo()
    gpus: list[GPUInfo] = []
    driver: str | None = None
    for row in rows.splitlines():
        columns = [column.strip() for column in row.split(",")]
        if len(columns) < 3:
            continue
        try:
            vram = int(float(columns[1])) * 1024 * 1024
        except ValueError:
            vram = None
        gpus.append(GPUInfo("NVIDIA", columns[0], vram))
        driver = driver or columns[2] or None
    summary = run_command(["nvidia-smi"])
    match = re.search(r"CUDA Version:\s*([0-9.]+)", summary or "")
    return NvidiaInfo(tuple(gpus), driver, match.group(1) if match else None)
