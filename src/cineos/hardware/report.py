"""Deterministic machine and human report formatting."""

from __future__ import annotations

import json

from .models import HardwareReport


def to_json(report: HardwareReport) -> str:
    return json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"


def to_text(report: HardwareReport, *, verbose: bool = False) -> str:
    value = report.to_dict()
    physical = report.physical_cpu_count or "unknown"
    logical = report.logical_cpu_count or "unknown"
    total_ram = report.total_ram_bytes or "unknown"
    available_ram = report.available_ram_bytes or "unknown"
    torch_cuda = (
        report.pytorch_cuda_available
        if report.pytorch_cuda_available is not None
        else "unknown"
    )
    lines = [
        "CINEOS Hardware Report",
        f"OS: {report.os} {report.os_version} ({report.architecture})",
        f"CPU: {report.cpu_model} ({physical} physical / {logical} logical)",
        f"RAM: total={total_ram} bytes, available={available_ram} bytes",
        f"GPUs: {report.gpu_count}",
    ]
    lines.extend(
        f"  - {gpu.vendor} {gpu.model}; VRAM={gpu.vram_bytes or 'unknown'} bytes"
        for gpu in report.gpus
    )
    lines.extend(
        [
            f"NVIDIA driver: {report.nvidia_driver_version or 'not detected'}",
            f"CUDA: {report.cuda_version or 'not detected'}",
            "PyTorch: "
            f"{'installed' if report.pytorch_installed else 'not installed'}; "
            f"CUDA available={torch_cuda}",
            f"FFmpeg: {report.ffmpeg_version or 'not detected'}",
            f"Free disk: {report.free_disk_bytes or 'unknown'} bytes",
            "Recommendation: "
            f"{report.recommendation.category} "
            f"({report.recommendation.renderer})",
            report.recommendation.guidance,
        ]
    )
    if verbose:
        lines.append("Raw deterministic data: " + json.dumps(value, sort_keys=True))
    return "\n".join(lines) + "\n"
