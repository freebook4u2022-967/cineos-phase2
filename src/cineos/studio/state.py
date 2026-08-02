"""Observable application state independent from Qt widgets."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cineos.core import MovieProject

from .models import RenderQueueItem, ReviewResult


@dataclass(slots=True)
class StudioState:
    project: MovieProject | None = None
    project_path: Path | None = None
    dirty: bool = False
    validation_errors: list[str] = field(default_factory=list)
    selected_scene_id: str | None = None
    selected_shot_id: str | None = None
    selected_asset_id: str | None = None
    selected_renderer: str = ""
    language: str = ""
    duration_target: float = 0.0
    queue: list[RenderQueueItem] = field(default_factory=list)
    reviews: dict[str, ReviewResult] = field(default_factory=dict)
    film_package: Any | None = None
    build: Any | None = None
    audio_project: Any | None = None
    audio_provider: str = ""
    audio_synthesis_progress: float = 0.0

    def require_project(self) -> MovieProject:
        if self.project is None:
            raise RuntimeError("no project is open")
        return self.project

    def mark_dirty(self) -> None:
        self.dirty = True

    @property
    def display_name(self) -> str:
        return self.project.title if self.project else "No project"
