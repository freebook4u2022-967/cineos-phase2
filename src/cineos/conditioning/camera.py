from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CameraConditioning:
    shot_type: str = ""
    framing: str = ""
    lens: str = ""
    aperture: Any = None
    camera_position: Any = None
    camera_movement: str = ""
    focus_target: Any = None
    depth_of_field_intent: Any = None
    aspect_ratio: str = ""
    resolution: tuple[int, int] = (1920, 1080)
    fps: float = 24.0
    duration: float = 0.0
