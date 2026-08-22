from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RenderContentStatus(StrEnum):
    VALID = "valid"
    BLACK_FRAME_FAILURE = "black_frame_failure"
    FROZEN_FRAME_FAILURE = "frozen_frame_failure"
    EMPTY_OUTPUT = "empty_output"
    DECODE_FAILURE = "decode_failure"
    RENDER_EXCEPTION = "render_exception"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"


@dataclass(slots=True)
class ShotRenderResult:
    shot_id: str
    output_file: str
    success: bool
    content_status: str
    file_size: int = 0
    duration: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    mean_luminance: float | None = None
    luminance_variance: float | None = None
    retry_attempted: bool = False
    retry_reason: str = ""
    original_settings: dict[str, Any] = field(default_factory=dict)
    retry_settings: dict[str, Any] = field(default_factory=dict)
    render_time: float = 0.0
    seed: int = 0
    model_id: str = ""
    gpu: str = ""
    vram_profile: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ColabRenderResult:
    project_id: str
    shots: list[dict] = field(default_factory=list)
    final_film: str = ""
    model_id: str = ""
    hardware: str = ""
    render_time_seconds: float = 0

    def to_dict(self):
        return asdict(self)
