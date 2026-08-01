"""Non-mutating environment inspection."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import LocalAIConfig


@dataclass(slots=True)
class EnvironmentReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, object] = field(default_factory=dict)


def validate_environment(config: LocalAIConfig) -> EnvironmentReport:
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, object] = {"os": platform.system(), "device": config.device}
    if platform.system() != "Linux":
        errors.append("local-ai supports Linux only")
    missing = [
        name
        for name in ("torch", "diffusers", "transformers", "accelerate")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        errors.append("missing Python dependencies: " + ", ".join(missing))
    if shutil.which("ffmpeg") is None:
        errors.append("FFmpeg executable is not available on PATH")
    model = Path(config.model_path).expanduser()
    if not model.is_dir() or not (model / "model_index.json").is_file():
        errors.append(f"missing model files: expected {model / 'model_index.json'}")
    output = Path(config.output_directory).expanduser()
    probe = output if output.exists() else output.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not os.access(probe, os.W_OK):
        errors.append(f"output directory is not writable: {output}")
    free = shutil.disk_usage(probe).free
    details["free_disk_bytes"] = free
    if free < config.minimum_disk_gb * 1024**3:
        errors.append(
            f"insufficient disk space: {free / 1024**3:.1f} GiB available, "
            f"{config.minimum_disk_gb:.1f} GiB required"
        )
    if config.device.startswith("cuda"):
        _validate_cuda(config, errors, details)
    else:
        warnings.append("CPU inference is supported but may take hours per short shot")
    return EnvironmentReport(not errors, errors, warnings, details)


def _validate_cuda(
    config: LocalAIConfig, errors: list[str], details: dict[str, object]
) -> None:
    if shutil.which("nvidia-smi") is None:
        errors.append("NVIDIA GPU/driver unavailable (nvidia-smi not found)")
        return
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or not result.stdout.strip():
        errors.append("unable to inspect NVIDIA GPU and driver")
        return
    memory, driver = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
    details.update(gpu_memory_mib=int(memory), nvidia_driver=driver)
    if int(memory) < config.minimum_vram_gb * 1024:
        errors.append(
            f"insufficient VRAM: {int(memory) / 1024:.1f} GiB available, "
            f"{config.minimum_vram_gb:.1f} GiB required"
        )
    try:
        import torch

        details["cuda_version"] = torch.version.cuda
        if not torch.cuda.is_available():
            errors.append(
                "PyTorch CUDA is unavailable or incompatible with the installed driver"
            )
    except ImportError:
        pass
