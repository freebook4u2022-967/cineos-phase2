"""Structured renderer result contract."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RenderResult:
    job_id: str
    shot_id: str
    renderer_id: str
    renderer_version: str
    model_identifier: str
    seed: int
    output_mp4_path: str
    duration: float
    resolution: tuple[int, int]
    fps: int
    render_time: float
    peak_vram_bytes: int | None
    warnings: tuple[str, ...]
    content_hash: str
    renderer_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
