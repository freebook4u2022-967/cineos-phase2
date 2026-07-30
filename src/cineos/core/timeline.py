"""Ordering and duration rules for project scenes and shots."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isclose

from .scene import Scene


@dataclass(slots=True)
class Timeline:
    """Store explicit scene order and per-scene shot order."""

    scene_order: list[str] = field(default_factory=list)
    shot_order: dict[str, list[str]] = field(default_factory=dict)

    def add_scene(self, scene_id: str, *, position: int | None = None) -> None:
        if scene_id in self.scene_order:
            raise ValueError(f"scene already exists on timeline: {scene_id}")
        if position is None:
            self.scene_order.append(scene_id)
        else:
            self.scene_order.insert(position, scene_id)
        self.shot_order.setdefault(scene_id, [])

    def add_shot(
        self, scene_id: str, shot_id: str, *, position: int | None = None
    ) -> None:
        if scene_id not in self.scene_order:
            raise KeyError(f"scene is not on timeline: {scene_id}")
        shots = self.shot_order.setdefault(scene_id, [])
        if shot_id in shots:
            raise ValueError(f"shot already exists on timeline: {shot_id}")
        if position is None:
            shots.append(shot_id)
        else:
            shots.insert(position, shot_id)

    def remove_scene(self, scene_id: str) -> None:
        self.scene_order.remove(scene_id)
        self.shot_order.pop(scene_id, None)

    def remove_shot(self, scene_id: str, shot_id: str) -> None:
        self.shot_order[scene_id].remove(shot_id)

    def validate_durations(
        self, scenes: Iterable[Scene], *, tolerance: float = 1e-6
    ) -> list[str]:
        """Report scenes whose declared duration differs from their shots."""

        errors: list[str] = []
        for scene in scenes:
            if not isclose(scene.duration, scene.shot_duration, abs_tol=tolerance):
                errors.append(
                    f"scene {scene.scene_id!r} duration {scene.duration} does not "
                    f"match shot duration {scene.shot_duration}"
                )
        return errors
