"""Renderer-neutral camera vocabulary."""

from dataclasses import dataclass
from enum import StrEnum


class Framing(StrEnum):
    ESTABLISHING = "establishing shot"
    WIDE = "wide shot"
    MEDIUM = "medium shot"
    CLOSE_UP = "close-up"
    EXTREME_CLOSE_UP = "extreme close-up"
    OVER_THE_SHOULDER = "over-the-shoulder"
    PROFILE = "profile"
    INSERT = "insert"
    POINT_OF_VIEW = "point-of-view"


class Angle(StrEnum):
    LOW = "low angle"
    HIGH = "high angle"
    EYE_LEVEL = "eye level"


class Movement(StrEnum):
    STATIC = "static"
    PAN = "pan"
    TILT = "tilt"
    DOLLY = "dolly"
    TRACKING = "tracking"
    PUSH_IN = "push-in"
    PULL_OUT = "pull-out"
    HANDHELD = "handheld"
    CRANE = "crane"
    AERIAL = "aerial"


@dataclass(slots=True)
class CameraPlan:
    framing: str = Framing.MEDIUM
    angle: str = Angle.EYE_LEVEL
    lens: str = "50mm"
    movement: str = Movement.STATIC
    focus_target: str = "principal action"


CameraLanguage = CameraPlan
