"""Studio orchestration layer delegating all domain work to existing APIs."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cineos.audio import AudioExporter, AudioValidator, ProviderRegistry
from cineos.audio.planner import plan_audio
from cineos.compiler import FilmCompiler
from cineos.compiler import save as save_package
from cineos.core import Character, Environment, MovieProject, ProjectValidator, Prop
from cineos.core.scene import Scene
from cineos.core.shot import Shot
from cineos.core.timeline import Timeline
from cineos.nova import (
    CreativeBrief,
    CritiqueFinding,
    NOVACritic,
    NOVADirector,
    NOVARevisionEngine,
)
from cineos.nova.director import DirectorPlan

from .state import StudioState

if TYPE_CHECKING:
    from .settings import StudioSettings


class StudioController:
    """Maintain UI state while using canonical CINEOS validation and compilation."""

    def __init__(
        self, state: StudioState | None = None, settings: StudioSettings | None = None
    ) -> None:
        self.state = state or StudioState()
        self.settings = settings
        self._listeners: list[Callable[[], None]] = []
        self.nova_plan: DirectorPlan | None = None

    def subscribe(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    def _changed(self) -> None:
        for listener in tuple(self._listeners):
            listener()

    def new_project(
        self, title: str = "Untitled Film", author: str = ""
    ) -> MovieProject:
        project = MovieProject(title=title, author=author)
        self.state = StudioState(project=project, dirty=True)
        self._changed()
        return project

    def update_metadata(self, **values: Any) -> None:
        project = self.state.require_project()
        for name in ("title", "author", "version", "fps", "resolution", "aspect_ratio"):
            if name in values:
                setattr(project, name, values[name])
        if "language" in values:
            self.state.language = str(values["language"])
        if "duration_target" in values:
            self.state.duration_target = float(values["duration_target"])
        self.state.mark_dirty()
        self._changed()

    def add_scene(self, scene: Scene) -> None:
        project = self.state.require_project()
        project.scenes.append(scene)
        project.timeline.add_scene(scene.scene_id)
        for shot in scene.shots:
            project.timeline.add_shot(scene.scene_id, shot.shot_id)
        scene.duration = scene.shot_duration
        self.state.mark_dirty()
        self._changed()

    def generate_nova_plan(
        self, brief: CreativeBrief, *, seed: int = 0, planner: str = "rule-based"
    ) -> DirectorPlan:
        """Generate story, scene, and shot plans while preserving approved assets."""
        registry = (
            self.state.project.asset_registry
            if self.state.project is not None
            else None
        )
        self.nova_plan = NOVADirector(registry).create_plan(
            brief, seed=seed, planner=planner
        )
        self.state.project = self.nova_plan.project
        self.state.mark_dirty()
        self._changed()
        return self.nova_plan

    def critique_nova_plan(self) -> list[CritiqueFinding]:
        if self.nova_plan is None:
            raise ValueError("generate a NOVA plan before critiquing it")
        return NOVACritic().critique(self.nova_plan)

    def revise_nova_plan(self, accepted: list[CritiqueFinding]) -> DirectorPlan:
        """Apply only accepted findings; rejected findings leave edits untouched."""
        if self.nova_plan is None:
            raise ValueError("generate a NOVA plan before revising it")
        self.nova_plan = NOVARevisionEngine().revise(self.nova_plan, accepted)
        self.state.project = self.nova_plan.project
        self.state.mark_dirty()
        self._changed()
        return self.nova_plan

    def add_shot(self, scene_id: str, shot: Shot) -> None:
        project = self.state.require_project()
        scene = next(item for item in project.scenes if item.scene_id == scene_id)
        scene.shots.append(shot)
        scene.duration = scene.shot_duration
        project.timeline.add_shot(scene_id, shot.shot_id)
        self.state.mark_dirty()
        self._changed()

    def move_scene(self, old: int, new: int) -> None:
        project = self.state.require_project()
        scene = project.scenes.pop(old)
        project.scenes.insert(new, scene)
        project.timeline.scene_order = [item.scene_id for item in project.scenes]
        self.state.mark_dirty()
        self._changed()

    def move_shot(self, scene_id: str, old: int, new: int) -> None:
        project = self.state.require_project()
        scene = next(item for item in project.scenes if item.scene_id == scene_id)
        shot = scene.shots.pop(old)
        scene.shots.insert(new, shot)
        project.timeline.shot_order[scene_id] = [item.shot_id for item in scene.shots]
        self.state.mark_dirty()
        self._changed()

    def validate(self) -> list[str]:
        self.state.validation_errors = ProjectValidator().validate(
            self.state.require_project()
        )
        self._changed()
        return self.state.validation_errors

    def compile(self) -> Any:
        self.state.film_package = FilmCompiler().compile(self.state.require_project())
        self._changed()
        return self.state.film_package

    def plan_audio(self, *, language: str | None = None) -> Any:
        """Create the Studio cue timeline while preserving existing dialogue edits."""
        package = self.state.film_package or self.compile()
        identifier = package.content_hashes.get("package", "")
        self.state.audio_project = plan_audio(
            self.state.require_project(),
            identifier,
            language=language or self.state.language or "en",
            existing=self.state.audio_project,
        )
        self._changed()
        return self.state.audio_project

    def select_audio_provider(self, provider_id: str) -> None:
        ProviderRegistry().get(provider_id)
        self.state.audio_provider = provider_id
        self._changed()

    def validate_audio(self) -> Any:
        project = self.state.audio_project or self.plan_audio()
        provider = (
            ProviderRegistry().get(self.state.audio_provider)
            if self.state.audio_provider
            else None
        )
        return AudioValidator().validate(project, provider=provider, check_ffmpeg=True)

    def export_audio(self, output_dir: str | Path) -> dict[str, Path]:
        project = self.state.audio_project or self.plan_audio()
        return AudioExporter().export(project, output_dir)

    def save_package(self, path: str | Path) -> None:
        if self.state.film_package is None:
            self.compile()
        save_package(self.state.film_package, path)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.state.project_path
        if target is None:
            raise ValueError("a destination is required")
        target.write_text(
            json.dumps(self._to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        self.state.project_path = target
        self.state.dirty = False
        if self.settings:
            self.settings.add_recent_project(str(target))
        self._changed()
        return target

    def open(self, path: str | Path) -> MovieProject:
        target = Path(path)
        data = json.loads(target.read_text(encoding="utf-8"))
        project = self._from_dict(data)
        self.state = StudioState(
            project=project,
            project_path=target,
            language=data.get("studio", {}).get("language", ""),
            duration_target=float(data.get("studio", {}).get("duration_target", 0)),
        )
        if self.settings:
            self.settings.add_recent_project(str(target))
        self._changed()
        return project

    def _to_dict(self) -> dict[str, Any]:
        project = self.state.require_project()
        return {
            "format": "cineos-project-1",
            "project": {
                "title": project.title,
                "author": project.author,
                "version": project.version,
                "fps": project.fps,
                "resolution": list(project.resolution),
                "aspect_ratio": project.aspect_ratio,
                "characters": [asdict(item) for item in project.characters],
                "locations": [asdict(item) for item in project.locations],
                "props": [asdict(item) for item in project.props],
                "scenes": [asdict(item) for item in project.scenes],
            },
            "studio": {
                "language": self.state.language,
                "duration_target": self.state.duration_target,
            },
        }

    @staticmethod
    def _from_dict(data: dict[str, Any]) -> MovieProject:
        item = data["project"]
        scenes = []
        timeline = Timeline()
        for raw_scene in item.get("scenes", []):
            shots = [Shot(**shot) for shot in raw_scene.get("shots", [])]
            scene = Scene(**{**raw_scene, "shots": shots})
            scenes.append(scene)
            timeline.add_scene(scene.scene_id)
            for shot in shots:
                timeline.add_shot(scene.scene_id, shot.shot_id)
        return MovieProject(
            title=item["title"],
            author=item.get("author", ""),
            version=item.get("version", "1.0"),
            fps=float(item.get("fps", 24)),
            resolution=tuple(item.get("resolution", [1920, 1080])),
            aspect_ratio=item.get("aspect_ratio", "16:9"),
            characters=[Character(**value) for value in item.get("characters", [])],
            locations=[Environment(**value) for value in item.get("locations", [])],
            props=[Prop(**value) for value in item.get("props", [])],
            scenes=scenes,
            timeline=timeline,
        )
