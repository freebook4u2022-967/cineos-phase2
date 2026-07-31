"""Hardware probing policy tests use fixed responses, never host capabilities."""

import importlib
import json

from cineos.hardware.models import GPUInfo
from cineos.hardware.nvidia import probe_nvidia
from cineos.hardware.recommendations import GIB, recommend
from cineos.hardware.report import to_json
from cineos.hardware.system import SystemInfo


def test_nvidia_probe_parses_multiple_gpus(monkeypatch) -> None:
    responses = iter(
        [
            "RTX 4090, 24564, 555.42\nRTX 3090, 24576, 555.42",
            "NVIDIA-SMI 555.42 CUDA Version: 12.5",
        ]
    )
    monkeypatch.setattr(
        "cineos.hardware.nvidia.run_command", lambda command: next(responses)
    )
    info = probe_nvidia()
    assert len(info.gpus) == 2
    assert info.gpus[0].vram_bytes == 24564 * 1024 * 1024
    assert info.driver_version == "555.42"
    assert info.cuda_version == "12.5"


def test_missing_nvidia_is_normal(monkeypatch) -> None:
    monkeypatch.setattr("cineos.hardware.nvidia.run_command", lambda command: None)
    assert probe_nvidia().gpus == ()


def test_recommendations_are_conservative() -> None:
    assert recommend((), 16 * GIB).category == "preview-only"
    assert recommend((), 2 * GIB).category == "unsupported"
    gpu = GPUInfo("NVIDIA", "GPU", 8 * GIB)
    assert recommend((gpu,), 16 * GIB).category == "standard-local"
    pair = (GPUInfo("AMD", "A", 4 * GIB), GPUInfo("AMD", "B", 4 * GIB))
    assert recommend(pair, 16 * GIB).category == "multi-gpu"


def test_report_json_is_deterministic(monkeypatch) -> None:
    implementation = importlib.import_module("cineos.hardware.probe")
    system = SystemInfo(
        "TestOS",
        "1",
        "x86_64",
        "3.12.0",
        "Test CPU",
        8,
        4,
        16 * GIB,
        8 * GIB,
        100 * GIB,
        None,
    )
    monkeypatch.setattr(implementation, "probe_system", lambda path=None: system)
    monkeypatch.setattr(implementation, "probe_other_gpus", lambda: ())
    monkeypatch.setattr(implementation, "probe_nvidia", lambda: probe_nvidia())
    monkeypatch.setattr("cineos.hardware.nvidia.run_command", lambda command: None)
    monkeypatch.setattr(implementation, "_torch_cuda", lambda: (False, None))
    report = implementation.probe()
    assert to_json(report) == to_json(report)
    payload = json.loads(to_json(report))
    assert payload["gpu_count"] == 0
    assert payload["ffmpeg_available"] is False
    assert payload["recommendation"]["category"] == "preview-only"
