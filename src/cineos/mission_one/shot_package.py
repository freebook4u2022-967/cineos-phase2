"""Compiled shot representation."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class DirectedShotPackage:
    shot_id: str
    duration: float
    prompt: str
    negative_prompt: str
    seed: int
    frame_count: int
    performance: dict[str, Any]
    dialogue: dict[str, Any]
    expected_output: str

    def to_dict(self):
        return asdict(self)
