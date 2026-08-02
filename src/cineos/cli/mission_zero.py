"""Auditable command implementation for the first real local-AI render."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from cineos.compiler import compile as compile_project
from cineos.compiler import save as save_film_package
from cineos.conditioning import (
    CameraConditioning,
    ConditioningPackage,
    ContinuityConditioning,
    RendererCapabilityRequirements,
)
from cineos.conditioning.serializer import save as save_conditioning
from cineos.renderers.local_ai import LocalAIConfig, LocalAIRendererPlugin
from cineos.renderers.local_ai.progress import RendererEvent

from .commands import load_project
from .errors import CLIError, ExitCode

BACKEND_ID = "damo-vilab/text-to-video-ms-1.7b"


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CLIError(f"cannot read {path}: {error}", code=ExitCode.INPUT) from error
    if not isinstance(value, dict):
        raise CLIError(f"{path} must contain a JSON object", code=ExitCode.INPUT)
    return value


def preflight(hardware_path: Path, config_path: Path) -> dict[str, Any]:
    """Validate a report captured on the target host; never probe cloud hardware."""
    report = _read_object(hardware_path)
    config = LocalAIConfig.load(config_path)
    required = (
        "os",
        "gpus",
        "nvidia_driver_version",
        "cuda_version",
        "available_ram_bytes",
        "free_disk_bytes",
        "ffmpeg_available",
        "python_version",
    )
    missing = [key for key in required if key not in report]
    failures: list[str] = []
    if missing:
        failures.append("report is missing fields: " + ", ".join(missing))
    gpus = report.get("gpus")
    gpu = gpus[0] if isinstance(gpus, list) and gpus else None
    vram = gpu.get("vram_bytes") if isinstance(gpu, dict) else None
    if report.get("os") != "Linux":
        failures.append("Linux is required")
    if not isinstance(gpu, dict):
        failures.append("an NVIDIA GPU is required")
    if not isinstance(vram, int):
        failures.append("GPU VRAM was not measured; capability will not be guessed")
    elif vram < int(config.minimum_vram_gb * 1024**3):
        failures.append(
            f"{config.minimum_vram_gb:.0f} GiB VRAM is required; "
            f"the report contains {vram / 1024**3:.1f} GiB"
        )
    for field, label in (
        ("nvidia_driver_version", "NVIDIA driver"),
        ("cuda_version", "CUDA support"),
    ):
        if not report.get(field):
            failures.append(f"{label} was not detected")
    if report.get("pytorch_cuda_available") is not True:
        failures.append("PyTorch CUDA support was not confirmed")
    if (report.get("available_ram_bytes") or 0) < 16 * 1024**3:
        failures.append("at least 16 GiB available RAM is required")
    if (report.get("free_disk_bytes") or 0) < config.minimum_disk_gb * 1024**3:
        failures.append(
            f"at least {config.minimum_disk_gb:.0f} GiB free disk is required"
        )
    if report.get("ffmpeg_available") is not True:
        failures.append("FFmpeg was not detected")
    if not str(report.get("python_version", "")).startswith("3.12"):
        failures.append("Python 3.12 is required by this project")
    model = Path(config.model_path).expanduser()
    if not model.is_absolute():
        model = (config_path.parent / model).resolve()
    if not (model / "model_index.json").is_file():
        failures.append(f"local model is missing {model / 'model_index.json'}")
    result = {
        "format": "cineos-mission-zero-preflight-v1",
        "passed": not failures,
        "backend": BACKEND_ID,
        "model_path": str(model),
        "hardware_report": str(hardware_path),
        "failures": failures,
        "upgrade_requirement": (
            None
            if not failures
            else "Linux NVIDIA CUDA workstation/cloud GPU with >=8 GiB VRAM, "
            ">=16 GiB available RAM, >=20 GiB free disk, Python 3.12, and FFmpeg"
        ),
    }
    return result


def render(project_path: Path, output_dir: Path) -> dict[str, Any]:
    """Compile, condition, execute Atlas/local-ai, and persist every boundary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = project_path.parent / "renderer-config.json"
    hardware_path = Path("hardware-report-local.json")
    check = preflight(hardware_path, config_path)
    (output_dir / "preflight-report.json").write_text(
        json.dumps(check, indent=2) + "\n", encoding="utf-8"
    )
    if not check["passed"]:
        raise CLIError(
            "Mission Zero preflight failed: " + "; ".join(check["failures"]),
            code=ExitCode.VALIDATION,
            hint="No inference was attempted and no success is being claimed.",
        )
    project = load_project(project_path)
    package = compile_project(project)
    shot = package.shot_manifest[0]
    shot_id = shot["shot_id"]
    config = LocalAIConfig.load(config_path)
    config.model_path = check["model_path"]
    config.output_directory = str(output_dir.resolve())
    conditioning = ConditioningPackage(
        shot_id,
        shot["scene_id"],
        [],
        None,
        [],
        [],
        CameraConditioning(
            resolution=(config.width, config.height),
            fps=config.fps,
            duration=config.duration,
        ),
        ContinuityConditioning(),
        [],
        RendererCapabilityRequirements(
            maximum_duration=config.duration,
            supported_resolution=(config.width, config.height),
            supported_fps=config.fps,
        ),
        config.seed,
    )
    (output_dir / "project.json").write_text(
        project_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    save_film_package(package, output_dir / "film-package.json")
    save_conditioning(conditioning, output_dir / "conditioning.json")
    events: list[dict[str, Any]] = []

    def record(event: RendererEvent) -> None:
        events.append({"name": event.name, "payload": event.payload})
        (output_dir / "runtime-log.json").write_text(
            json.dumps(events, indent=2, default=str) + "\n", encoding="utf-8"
        )

    plugin = LocalAIRendererPlugin(config, event_sink=record)
    request = plugin.create_request(
        package, conditioning, output_dir / "shot-001.mp4", shot_id=shot_id
    )
    request_data = {
        "renderer_id": "local-ai",
        "model": BACKEND_ID,
        "shot_id": request.shot_id,
        "prompt": request.prompt,
        "seed": request.seed,
        "resolution": [request.width, request.height],
        "fps": request.fps,
        "duration": request.duration,
    }
    (output_dir / "renderer-request.json").write_text(
        json.dumps(request_data, indent=2) + "\n", encoding="utf-8"
    )
    result = plugin.render(
        package, conditioning, output_dir / "shot-001.mp4", shot_id=shot_id
    )
    result_data = result.to_dict()
    (output_dir / "render-result.json").write_text(
        json.dumps(result_data, indent=2) + "\n", encoding="utf-8"
    )
    return result_data


def verify(path: Path) -> dict[str, Any]:
    failures: list[str] = []
    if not path.is_file() or path.stat().st_size == 0:
        failures.append("output MP4 does not exist or is empty")
    if shutil.which("ffprobe") is None:
        failures.append("ffprobe is unavailable")
        probe: dict[str, Any] = {}
    else:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_name,width,height,avg_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            probe = json.loads(completed.stdout) if completed.returncode == 0 else {}
        except json.JSONDecodeError:
            probe = {}
        streams = probe.get("streams", [])
        if not streams or streams[0].get("codec_name") not in {"h264", "hevc", "mpeg4"}:
            failures.append("no playable MP4 video stream was found")
        if float(probe.get("format", {}).get("duration", 0) or 0) <= 0:
            failures.append("video duration is zero")
    report = {
        "format": "cineos-mission-zero-verification-v1",
        "passed": not failures,
        "output": str(path),
        "sha256": (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        ),
        "probe": probe,
        "failures": failures,
    }
    destination = path.parent / "verification-report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
