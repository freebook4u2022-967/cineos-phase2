"""Portable system probes with optional platform-specific improvements."""

from __future__ import annotations

import os
import platform
import plistlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import GPUInfo


@dataclass(frozen=True, slots=True)
class SystemInfo:
    os: str
    os_version: str
    architecture: str
    python_version: str
    cpu_model: str
    logical_cpu_count: int | None
    physical_cpu_count: int | None
    total_ram_bytes: int | None
    available_ram_bytes: int | None
    free_disk_bytes: int | None
    ffmpeg_version: str | None


def run_command(command: list[str], timeout: float = 5.0) -> str | None:
    """Run a diagnostic executable without a shell and absorb expected failures."""

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def _psutil_values() -> tuple[int | None, int | None, int | None]:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None, None, None
    memory = psutil.virtual_memory()
    return psutil.cpu_count(logical=False), int(memory.total), int(memory.available)


def _posix_memory() -> tuple[int | None, int | None]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = page_size * os.sysconf("SC_PHYS_PAGES")
        available = page_size * os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None, None
    return int(total), int(available)


def _cpu_model() -> str:
    model = platform.processor().strip()
    if model:
        return model
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.partition(":")[2].strip()
        except OSError:
            pass
    return "unknown"


def _ffmpeg_version() -> str | None:
    output = run_command(["ffmpeg", "-version"])
    return output.splitlines()[0] if output else None


def probe_other_gpus() -> tuple[GPUInfo, ...]:
    """Best-effort discovery for non-NVIDIA adapters on each supported OS."""

    system = platform.system()
    if system == "Darwin":
        output = run_command(["system_profiler", "SPDisplaysDataType", "-xml"])
        if not output:
            return ()
        try:
            documents = plistlib.loads(output.encode())
            items = documents[0]["_items"]
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            plistlib.InvalidFileException,
        ):
            return ()
        return tuple(
            GPUInfo(
                "Apple" if "Apple" in item.get("sppci_model", "") else "unknown",
                item.get("sppci_model", "unknown"),
            )
            for item in items
        )
    if system == "Windows":
        output = run_command(
            ["wmic", "path", "win32_VideoController", "get", "name", "/value"]
        )
        names = (
            [line.partition("=")[2].strip() for line in output.splitlines()]
            if output
            else []
        )
    elif system == "Linux":
        output = run_command(["lspci", "-mm"])
        names = [
            line
            for line in (output or "").splitlines()
            if "VGA compatible controller" in line or "3D controller" in line
        ]
    else:
        names = []
    results = []
    for name in names:
        lowered = name.lower()
        vendor = next(
            (
                item
                for item in ("NVIDIA", "AMD", "Intel", "Apple")
                if item.lower() in lowered
            ),
            "unknown",
        )
        if vendor != "NVIDIA":  # NVIDIA has richer, deduplicated nvidia-smi data.
            results.append(GPUInfo(vendor, name))
    return tuple(results)


def probe_system(path: Path | None = None) -> SystemInfo:
    """Collect OS, processor, memory, disk, and FFmpeg information."""

    physical, total, available = _psutil_values()
    if total is None:
        total, available = _posix_memory()
    try:
        free_disk = shutil.disk_usage(path or Path.cwd()).free
    except OSError:
        free_disk = None
    return SystemInfo(
        os=platform.system() or "unknown",
        os_version=platform.version() or platform.release() or "unknown",
        architecture=platform.machine() or "unknown",
        python_version=platform.python_version(),
        cpu_model=_cpu_model(),
        logical_cpu_count=os.cpu_count(),
        physical_cpu_count=physical,
        total_ram_bytes=total,
        available_ram_bytes=available,
        free_disk_bytes=free_disk,
        ffmpeg_version=_ffmpeg_version(),
    )
