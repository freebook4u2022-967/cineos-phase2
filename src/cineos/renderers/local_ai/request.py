"""Deterministic backend request construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RenderRequest:
    job_id: str
    shot_id: str
    prompt: str
    seed: int
    output_path: Path
    width: int
    height: int
    fps: int
    duration: float
    inference_steps: int
    guidance: float
    approved_reference_ids: tuple[str, ...]
    cinedna_ids: tuple[str, ...]
    metadata: Mapping[str, Any]

    @property
    def frame_count(self) -> int:
        return max(1, round(self.fps * self.duration))


def build_prompt(shot: Mapping[str, Any]) -> str:
    parts = []
    for key in ("description", "action", "prompt", "camera", "lighting"):
        value = shot.get(key)
        if value:
            parts.append(str(value).strip())
    return ", ".join(parts) or f"cinematic shot {shot.get('shot_id', '')}".strip()
