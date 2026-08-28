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
    details: dict[str, object] = {
        "os": platform.system(),
        "device": config.device,
        "model_source": "remote" if config.allow_remote_model else "local",
        "production_mode": config.production_mode,
    }
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
    has_local_model = model.is_dir() and (model / "model_index.json").is_file()
    if not has_local_model:
        if not config.allow_remote_model:
            errors.append(f"missing model files: expected {model / 'model_index.json'}")
        else:
            if not config.model_license or not config.model_license.strip():
                errors.append(
                    "remote pretrained models require model_license to be declared"
                )
            if not config.model_provenance or not config.model_provenance.strip():
                errors.append(
                    "remote pretrained models require model_provenance to be declared"
                )
            if config.production_mode and not config.model_revision:
                errors.append(
                    "production remote models require model_revision to be pinned"
                )
            details["remote_model_id"] = config.model_path
            if config.model_revision:
                details["remote_model_revision"] = config.model_revision
            warnings.append(
                "remote model download is enabled; verify license terms and pin a revision "
                "for reproducible production runs"
            )

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
    if config.production_mode and not config.device.startswith("cuda"):
        errors.append("production_mode requires CUDA-backed inference")
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
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or not result.stdout.strip():
        errors.append("unable to inspect NVIDIA GPU and driver")
        return
    gpu_name, memory, driver = [
        part.strip() for part in result.stdout.splitlines()[0].split(",", 2)
    ]
    details.update(
        gpu_name=gpu_name,
        gpu_memory_mib=int(memory),
        nvidia_driver=driver,
    )
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
        else:
            device_index = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device_index)
            details["cuda_compute_capability"] = f"{capability[0]}.{capability[1]}"
            details["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except ImportError:
        pass
