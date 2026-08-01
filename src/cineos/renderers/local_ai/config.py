"""Configuration for the single local AI backend."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LocalAIConfig:
    model_path: str = "models/text-to-video-ms-1.7b"
    device: str = "cpu"
    precision: str = "float32"
    width: int = 576
    height: int = 320
    fps: int = 8
    duration: float = 2.0
    inference_steps: int = 25
    guidance: float = 9.0
    seed: int = 0
    output_format: str = "mp4"
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    cpu_offload: bool = False
    minimum_vram_gb: float = 8.0
    minimum_disk_gb: float = 5.0
    output_directory: str = "renders"

    @classmethod
    def load(cls, path: str | Path | None = None) -> LocalAIConfig:
        source = Path(path or "renderer.local-ai.json")
        if not source.exists():
            return cls()
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("renderer config must contain a JSON object")
        unknown = set(value) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(
                f"unknown renderer config keys: {', '.join(sorted(unknown))}"
            )
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
