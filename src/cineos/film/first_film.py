"""Fast-track end-to-end FIRST FILM coordinator.

This module intentionally keeps the critical path small: a premise becomes a
renderable shot package, continuity is locked through stable character IDs, the
existing FilmOrchestrator performs render/QC/retry/assembly, and the result is a
single FilmBuild with an explicit final MP4 path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .audio import AudioTrack, mux_primary_audio
from .build import BuildStatus, FilmBuild
from .orchestrator import FilmOrchestrator


@dataclass(frozen=True, slots=True)
class DirectorCharacter:
    character_id: str
    name: str
    reference_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FirstFilmPackage:
    package_id: str
    premise: str
    shot_manifest: tuple[dict[str, Any], ...]
    timeline_manifest: dict[str, Any]
    character_manifest: tuple[dict[str, Any], ...]


class FastTrackAutoDirector:
    """Deterministic minimum viable director for FIRST FILM validation.

    It creates a three-beat short-drama grammar (setup, escalation, payoff)
    while preserving stable character identifiers across every shot. A future
    story model can replace this director without changing the renderer/QC path.
    """

    def __init__(self, *, shot_duration: float = 4.0) -> None:
        if shot_duration <= 0:
            raise ValueError("shot_duration must be positive")
        self.shot_duration = float(shot_duration)

    def direct(
        self,
        premise: str,
        characters: Iterable[DirectorCharacter],
        *,
        package_id: str = "first-film",
    ) -> FirstFilmPackage:
        text = premise.strip()
        cast = tuple(characters)
        if not text:
            raise ValueError("premise must not be empty")
        if not cast:
            raise ValueError("at least one character is required")
        ids = [item.character_id.strip() for item in cast]
        if any(not item for item in ids) or len(set(ids)) != len(ids):
            raise ValueError("character IDs must be non-empty and unique")

        beats = (
            ("setup", "Establish the world, protagonist and immediate objective."),
            ("escalation", "Introduce a visible obstacle and raise urgency."),
            ("payoff", "Resolve the immediate dramatic question with a strong final image."),
        )
        shots: list[dict[str, Any]] = []
        order: list[str] = []
        for index, (beat, instruction) in enumerate(beats, start=1):
            shot_id = f"shot-{index:03d}"
            order.append(shot_id)
            shots.append(
                {
                    "shot_id": shot_id,
                    "scene_id": "scene-001",
                    "duration": self.shot_duration,
                    "beat": beat,
                    "prompt": f"{text} {instruction}",
                    "character_ids": tuple(ids),
                    "continuity_key": "scene-001:" + ":".join(ids),
                }
            )
        return FirstFilmPackage(
            package_id=package_id,
            premise=text,
            shot_manifest=tuple(shots),
            timeline_manifest={
                "scene_order": ["scene-001"],
                "shot_order": {"scene-001": order},
            },
            character_manifest=tuple(
                {
                    "character_id": item.character_id,
                    "name": item.name,
                    "reference_paths": item.reference_paths,
                }
                for item in cast
            ),
        )


class FirstFilmRunner:
    """One-call Auto Director -> render -> QC/retry -> audio -> final MP4 runner."""

    def __init__(
        self,
        renderer: Any,
        validator: Any | None = None,
        *,
        renderer_id: str = "atlas",
        max_recovery_attempts: int = 2,
    ) -> None:
        self.renderer = renderer
        self.renderer_id = renderer_id
        self.orchestrator = FilmOrchestrator(
            renderer,
            validator,
            max_recovery_attempts=max_recovery_attempts,
            manual_review_on_failure=False,
        )

    def run(
        self,
        premise: str,
        characters: Iterable[DirectorCharacter],
        output_dir: str | Path,
        *,
        project_id: str = "cineos-first-film",
        package_id: str = "first-film",
        shot_duration: float = 4.0,
        audio_tracks: list[AudioTrack] | None = None,
        dry_run: bool = False,
    ) -> FilmBuild:
        package = FastTrackAutoDirector(shot_duration=shot_duration).direct(
            premise, characters, package_id=package_id
        )
        build = FilmBuild(
            project_id=project_id,
            film_package_id=package.package_id,
            renderer_id=self.renderer_id,
        )
        build.metadata["first_film"] = {
            "premise": package.premise,
            "character_ids": [item["character_id"] for item in package.character_manifest],
            "critical_path": [
                "auto_director",
                "continuity_lock",
                "renderer",
                "qc_retry",
                "assembly",
                "audio_mux",
            ],
        }
        result = self.orchestrator.run(package, build, output_dir, dry_run=dry_run)
        if dry_run or result.status not in {
            BuildStatus.COMPLETED,
            BuildStatus.COMPLETED_WITH_WARNINGS,
        }:
            return result

        root = Path(output_dir)
        video = result.output_files.get("final_mp4")
        if not video:
            result.failures.append("final assembly did not produce final_mp4")
            result.transition(BuildStatus.FAILED)
            return result
        final_with_audio = mux_primary_audio(video, audio_tracks, root / "first-film.mp4")
        result.output_files["final_mp4"] = str(final_with_audio)
        result.attach_audio(
            str(audio_tracks[0].path) if audio_tracks and audio_tracks[0].path.is_file() else None,
            {"silent_fallback": not bool(audio_tracks), "track_count": len(audio_tracks or [])},
        )
        return result
