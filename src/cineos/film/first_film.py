"""End-to-end FIRST FILM coordinator.

A premise becomes a renderable shot package, continuity is locked through stable
character IDs, the existing FilmOrchestrator performs render/QC/retry/assembly,
and the result is a single FilmBuild with an explicit final MP4 path. Production
callers may additionally inject a measured final-film evaluator so the fully muxed
movie is accepted only after post-assembly temporal/edit QC.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .audio import AudioTrack, available_tracks, mux_audio_tracks
from .build import BuildStatus, FilmBuild
from .orchestrator import FilmOrchestrator
from .planner import plan_shots

FIRST_FILM_RESUME_CONTRACT_SCHEMA = "cineos-first-film-resume/0.1"


class FinalFilmEvaluator(Protocol):
    """Renderer-neutral post-assembly quality evaluator contract."""

    def evaluate(self, movie_path: str | Path, plan: Any) -> Any: ...


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
    """Deterministic short-drama director for connected-film validation.

    The default three-shot grammar is preserved for backwards compatibility, while
    production benchmark callers can request 5-10 connected shots. Every generated
    shot preserves the same stable character identifiers and continuity key so the
    renderer/QC path can exercise temporal and identity persistence across a longer
    film instead of validating only isolated clips.
    """

    _BEATS = (
        ("setup", "Establish the world, protagonist and immediate objective."),
        (
            "inciting_action",
            "Trigger a visible event that forces the protagonist to act.",
        ),
        ("escalation", "Introduce a visible obstacle and raise urgency."),
        (
            "interaction",
            "Force a character or object interaction that changes the situation.",
        ),
        ("movement", "Advance the action through purposeful walking or running."),
        (
            "reversal",
            "Reveal a setback or reversal while preserving spatial continuity.",
        ),
        (
            "pressure",
            "Increase pressure with stronger movement, blocking or camera energy.",
        ),
        ("choice", "Make the protagonist perform a clear consequential choice."),
        (
            "resolution",
            "Resolve the immediate dramatic conflict through visible action.",
        ),
        ("payoff", "End on a strong final image that preserves character identity."),
    )

    def __init__(self, *, shot_duration: float = 4.0, shot_count: int = 3) -> None:
        if shot_duration <= 0:
            raise ValueError("shot_duration must be positive")
        if isinstance(shot_count, bool) or not isinstance(shot_count, int):
            raise ValueError("shot_count must be an integer")
        if shot_count < 3 or shot_count > len(self._BEATS):
            raise ValueError("shot_count must be between 3 and 10")
        self.shot_duration = float(shot_duration)
        self.shot_count = shot_count

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

        beats = self._selected_beats()
        shots: list[dict[str, Any]] = []
        order: list[str] = []
        continuity_key = "scene-001:" + ":".join(ids)
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
                    "continuity_key": continuity_key,
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

    def _selected_beats(self) -> tuple[tuple[str, str], ...]:
        """Return a coherent arc while preserving the historical 3-shot grammar."""
        if self.shot_count == 3:
            return (self._BEATS[0], self._BEATS[2], self._BEATS[-1])
        if self.shot_count == len(self._BEATS):
            return self._BEATS

        # Keep setup/payoff fixed and distribute intermediate dramatic challenges
        # over the available grammar without duplicate beats.
        interior = self._BEATS[1:-1]
        needed = self.shot_count - 2
        if needed == 1:
            selected = (interior[len(interior) // 2],)
        else:
            last = len(interior) - 1
            indices = tuple(
                round(index * last / (needed - 1)) for index in range(needed)
            )
            selected = tuple(interior[index] for index in indices)
        return (self._BEATS[0], *selected, self._BEATS[-1])


def _resume_contract(
    package: FirstFilmPackage,
    *,
    renderer_id: str,
    require_final_film_evaluation: bool,
) -> dict[str, str]:
    """Fingerprint all render-significant FIRST FILM intent for safe resume."""
    payload = {
        "schema": FIRST_FILM_RESUME_CONTRACT_SCHEMA,
        "renderer_id": renderer_id,
        "premise": package.premise,
        "shot_manifest": package.shot_manifest,
        "timeline_manifest": package.timeline_manifest,
        "character_manifest": package.character_manifest,
        "final_film_qc_required": require_final_film_evaluation,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "schema": FIRST_FILM_RESUME_CONTRACT_SCHEMA,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


class FirstFilmRunner:
    """One-call Auto Director -> render -> QC/retry -> audio -> final-film QC runner.

    ``orchestrator_kwargs`` is the provider-neutral extension boundary for durable
    runtime state and transactional shot lifecycle hooks. Native temporal video can
    pass ``NativeFilmContinuityBridge.orchestrator_kwargs()`` here without coupling
    the film layer to model-specific tensors or devices.

    ``final_film_evaluator`` is intentionally another neutral boundary. Production
    native video can inject ``MeasuredFinalFilmGate`` to evaluate decoded pixels of
    the *fully muxed movie*. ``require_final_film_evaluation`` fails closed when a
    production caller accidentally omits that gate, while remaining opt-in for
    backwards compatibility with older integrations and lightweight test renderers.
    """

    def __init__(
        self,
        renderer: Any,
        validator: Any | None = None,
        *,
        renderer_id: str = "atlas",
        max_recovery_attempts: int = 2,
        orchestrator_kwargs: Mapping[str, Any] | None = None,
        final_film_evaluator: FinalFilmEvaluator | None = None,
        require_final_film_evaluation: bool = False,
    ) -> None:
        if require_final_film_evaluation and final_film_evaluator is None:
            raise ValueError("production FIRST FILM requires a final_film_evaluator")
        self.renderer = renderer
        self.renderer_id = renderer_id
        self.final_film_evaluator = final_film_evaluator
        self.require_final_film_evaluation = require_final_film_evaluation
        runtime_hooks = dict(orchestrator_kwargs or {})
        reserved = {"max_recovery_attempts", "manual_review_on_failure"}
        conflicts = sorted(reserved.intersection(runtime_hooks))
        if conflicts:
            joined = ", ".join(conflicts)
            raise ValueError(
                f"orchestrator_kwargs cannot override runner policy: {joined}"
            )
        self.orchestrator = FilmOrchestrator(
            renderer,
            validator,
            max_recovery_attempts=max_recovery_attempts,
            manual_review_on_failure=False,
            **runtime_hooks,
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
        shot_count: int = 3,
        audio_tracks: list[AudioTrack] | None = None,
        dry_run: bool = False,
        resume: bool = False,
        checkpoint_path: str | Path | None = None,
    ) -> FilmBuild:
        package = FastTrackAutoDirector(
            shot_duration=shot_duration,
            shot_count=shot_count,
        ).direct(premise, characters, package_id=package_id)
        build = FilmBuild(
            project_id=project_id,
            film_package_id=package.package_id,
            renderer_id=self.renderer_id,
        )
        build.metadata["resume_contract"] = _resume_contract(
            package,
            renderer_id=self.renderer_id,
            require_final_film_evaluation=self.require_final_film_evaluation,
        )
        build.metadata["first_film"] = {
            "premise": package.premise,
            "character_ids": [
                item["character_id"] for item in package.character_manifest
            ],
            "shot_count": len(package.shot_manifest),
            "critical_path": [
                "auto_director",
                "continuity_lock",
                "renderer",
                "qc_retry",
                "assembly",
                "audio_mux",
                "final_film_qc",
            ],
            "runtime_checkpointing": checkpoint_path is not None,
            "final_film_qc_required": self.require_final_film_evaluation,
            "final_film_qc_enabled": self.final_film_evaluator is not None,
        }
        result = self.orchestrator.run(
            package,
            build,
            output_dir,
            dry_run=dry_run,
            resume=resume,
            checkpoint_path=checkpoint_path,
        )
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
        usable_audio = available_tracks(audio_tracks)
        final_with_audio = mux_audio_tracks(
            video,
            usable_audio,
            root / "first-film.mp4",
        )
        result.output_files["final_mp4"] = str(final_with_audio)
        result.attach_audio(
            str(usable_audio[0].path) if usable_audio else None,
            {
                "silent_fallback": not bool(usable_audio),
                "track_count": len(usable_audio),
                "mix_mode": "timeline_multitrack" if usable_audio else "silent",
                "kinds": [track.kind for track in usable_audio],
            },
        )
        self._evaluate_final_movie(result, final_with_audio, package)
        return result

    def _evaluate_final_movie(
        self,
        build: FilmBuild,
        movie_path: str | Path,
        package: FirstFilmPackage,
    ) -> None:
        evaluator = self.final_film_evaluator
        if evaluator is None:
            build.metadata["final_film_qc"] = {
                "enabled": False,
                "required": self.require_final_film_evaluation,
            }
            return

        report = evaluator.evaluate(movie_path, plan_shots(package))
        decision = str(getattr(report, "decision", "")).strip().lower()
        if decision not in {"accept", "warn", "reject"}:
            build.failures.append(
                "final-film evaluator returned an invalid decision; expected "
                "accept, warn, or reject"
            )
            build.metadata["final_film_qc"] = {
                "enabled": True,
                "required": self.require_final_film_evaluation,
                "decision": decision or None,
            }
            build.transition(BuildStatus.FAILED)
            return

        as_dict = getattr(report, "as_dict", None)
        evidence = as_dict() if callable(as_dict) else {"decision": decision}
        build.metadata["final_film_qc"] = {
            "enabled": True,
            "required": self.require_final_film_evaluation,
            "decision": decision,
            "evidence": evidence,
        }
        directives = tuple(getattr(report, "directives", ()) or ())
        if decision == "reject":
            detail = (
                "; ".join(str(item) for item in directives)
                or "quality gate rejected movie"
            )
            build.failures.append(f"final-film QC rejected assembled movie: {detail}")
            build.transition(BuildStatus.FAILED)
        elif decision == "warn":
            detail = (
                "; ".join(str(item) for item in directives) or "quality gate warning"
            )
            build.warnings.append(f"final-film QC warning: {detail}")
            build.transition(BuildStatus.COMPLETED_WITH_WARNINGS)
