"""End-to-end FIRST FILM path using the full CINEOS Short Drama brains."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cineos.film.build import FilmBuild
from cineos.film.orchestrator import FilmOrchestrator

from .integration import compile_drama_plan
from .models import DramaBrief
from .orchestrator import ShortDramaOrchestrator


class ShortDramaFirstFilmRunner:
    """Premise -> CINEOS brains/director -> FilmPackage -> render/QC/assembly."""

    def __init__(
        self,
        renderer: Any,
        validator: Any | None = None,
        *,
        renderer_id: str = "atlas",
        max_recovery_attempts: int = 2,
    ) -> None:
        self.renderer_id = renderer_id
        self.director = ShortDramaOrchestrator()
        self.film = FilmOrchestrator(
            renderer,
            validator,
            max_recovery_attempts=max_recovery_attempts,
            manual_review_on_failure=False,
        )

    def run(
        self,
        premise: str,
        output_dir: str | Path,
        *,
        duration_seconds: int = 30,
        genre: str = "drama",
        tone: str = "cinematic",
        dry_run: bool = False,
    ) -> FilmBuild:
        brief = DramaBrief(
            premise=premise,
            duration_seconds=duration_seconds,
            genre=genre,
            tone=tone,
        )
        plan = self.director.plan(brief)
        project, package = compile_drama_plan(plan)
        build = FilmBuild(
            project_id=project.title,
            film_package_id=package.content_hashes.get("package", "short-drama"),
            renderer_id=self.renderer_id,
        )
        build.metadata["auto_director"] = {
            "engine": "ShortDramaOrchestrator",
            "continuity_status": plan.continuity.get("status"),
            "scene_count": len(plan.screenplay.get("scenes", ())),
            "shot_count": len(plan.shots),
            "critical_path": [
                "drama_brain",
                "character_brain",
                "screenwriter",
                "director",
                "shot_planner",
                "continuity_supervisor",
                "film_compiler",
                "renderer",
                "qc_retry",
                "assembly",
            ],
        }
        return self.film.run(package, build, output_dir, dry_run=dry_run)
