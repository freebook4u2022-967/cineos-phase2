"""Deterministic film-aligned audio timeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dialogue import DialogueCue


@dataclass(frozen=True, slots=True)
class TimingRange:
    start: float
    end: float


@dataclass(slots=True)
class AudioTimeline:
    scene_timing: dict[str, TimingRange] = field(default_factory=dict)
    shot_timing: dict[str, TimingRange] = field(default_factory=dict)
    cues: list[Any] = field(default_factory=list)
    gain_automation: dict[str, list[dict[str, float]]] = field(default_factory=dict)

    def align_scene(self, scene_id: str, start: float, duration: float) -> None:
        self.scene_timing[scene_id] = TimingRange(start, start + duration)

    def align_shot(
        self, scene_id: str, shot_id: str, start: float, duration: float
    ) -> None:
        scene = self.scene_timing.get(scene_id)
        if scene and (start < scene.start or start + duration > scene.end + 1e-9):
            raise ValueError(f"shot {shot_id} falls outside scene {scene_id}")
        self.shot_timing[shot_id] = TimingRange(start, start + duration)

    def add_cue(self, cue: Any) -> None:
        shot_id = getattr(cue, "shot_id", None)
        span = (
            self.shot_timing.get(shot_id)
            if shot_id
            else self.scene_timing.get(cue.scene_id)
        )
        end = cue.start_time + getattr(cue, "duration", 0)
        if span and (cue.start_time < span.start or end > span.end + 1e-9):
            raise ValueError(f"cue {cue.cue_id} is not aligned to film timing")
        self.cues.append(cue)
        self.cues.sort(
            key=lambda item: (
                item.start_time,
                item.scene_id,
                getattr(item, "shot_id", "") or "",
                item.cue_id,
            )
        )

    def dialogue(self, language: str | None = None) -> list[DialogueCue]:
        result = [item for item in self.cues if isinstance(item, DialogueCue)]
        return [
            item for item in result if language is None or item.language == language
        ]

    def overlap_conflicts(self) -> list[tuple[str, str]]:
        conflicts = []
        active = [item for item in self.cues if not getattr(item, "muted", False)]
        for index, left in enumerate(active):
            left_end = left.start_time + getattr(left, "duration", 0)
            for right in active[index + 1 :]:
                if right.start_time >= left_end:
                    break
                if not isinstance(left, DialogueCue) or not isinstance(
                    right, DialogueCue
                ):
                    continue
                if "allow" not in (left.overlap_rules, right.overlap_rules):
                    conflicts.append((left.cue_id, right.cue_id))
        return conflicts

    @property
    def duration(self) -> float:
        spans = [item.end for item in self.scene_timing.values()]
        spans += [item.start_time + getattr(item, "duration", 0) for item in self.cues]
        return max(spans, default=0.0)
