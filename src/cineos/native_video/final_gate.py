"""Production final-film quality gate for the CINEOS FIRST FILM path.

This module adapts native video evaluators to the provider-neutral ``FirstFilmRunner``
contract. It combines whole-film temporal evidence, plan-aware scene-boundary
evidence, encoded-duration integrity, and optional measured audio integrity. It fails
closed when measured quality or assembly completeness is rejected. External tools
are inspectors/decoders only; no external visual generator participates in acceptance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audio_integrity import AudioIntegrityReport, FinalFilmAudioIntegrityGate
from .boundary_eval import FFmpegSceneBoundaryEvaluator, SceneBoundaryPoint
from .duration_gate import DurationIntegrityReport, FFprobeDurationIntegrityGate
from .final_eval import (
    FFmpegTemporalFilmEvaluator,
    SceneBoundaryEvalReport,
    TemporalFilmEvalReport,
)


@dataclass(frozen=True, slots=True)
class MeasuredFinalFilmReport:
    """Auditable aggregate of final-film picture, assembly, and audio evidence."""

    decision: str
    directives: tuple[str, ...]
    temporal: TemporalFilmEvalReport
    boundaries: SceneBoundaryEvalReport | None = None
    duration: DurationIntegrityReport | None = None
    audio: AudioIntegrityReport | None = None

    @property
    def accepted(self) -> bool:
        return self.decision in {"accept", "warn"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "directives": list(self.directives),
            "temporal": asdict(self.temporal),
            "boundaries": (
                asdict(self.boundaries) if self.boundaries is not None else None
            ),
            "duration": asdict(self.duration) if self.duration is not None else None,
            "audio": asdict(self.audio) if self.audio is not None else None,
        }


def _shot_payload(shot: Any) -> Mapping[str, Any]:
    payload = getattr(shot, "payload", None)
    return payload if isinstance(payload, Mapping) else {}


def _metadata_flag(payload: Mapping[str, Any], name: str) -> bool:
    """Decode persisted boolean edit metadata without truthiness ambiguity.

    Shot plans routinely cross JSON/YAML/CLI boundaries. Values such as ``"false"``
    must not become true merely because they are non-empty strings. Production
    acceptance therefore accepts conventional serialized boolean forms and rejects
    ambiguous values instead of silently inventing edit semantics.
    """

    raw = payload.get(name, False)
    if raw is None:
        return False
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int) and raw in {0, 1}:
        return bool(raw)
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise ValueError(f"{name} must be boolean metadata; got {raw!r}")


def _planned_transition(previous_shot: Any, shot: Any) -> str:
    """Resolve the authored scene-transition contract deterministically.

    Explicit continuity resets and hard cuts on the incoming shot take precedence
    over transition hints. This is critical for long-form production: a scene reset
    must never be evaluated as a match/fade simply because stale transition metadata
    survived from a prior planning pass. Legacy ``transition_in``/``transition_out``
    keys remain supported alongside the newer generic transition vocabulary.
    """

    payload = _shot_payload(shot)
    previous_payload = _shot_payload(previous_shot)
    if _metadata_flag(payload, "continuity_reset") or _metadata_flag(
        payload, "hard_cut"
    ):
        return "cut"

    raw = payload.get("transition_in")
    if raw is None:
        raw = payload.get("scene_transition")
    if raw is None:
        raw = payload.get("transition")
    if raw is None:
        raw = previous_payload.get("transition_out")
    if raw is None:
        raw = previous_payload.get("scene_transition")
    if raw is None:
        raw = previous_payload.get("transition")
    if raw is None:
        raw = "cut"

    transition = str(raw).strip().lower()
    aliases = {
        "hard_cut": "cut",
        "hard-cut": "cut",
        "crossfade": "fade",
        "cross_fade": "fade",
        "match_cut": "match",
        "match-cut": "match",
    }
    transition = aliases.get(transition, transition)
    if transition not in {"cut", "match", "fade"}:
        raise ValueError(
            f"unsupported planned scene transition {raw!r}; "
            "expected cut, match, or fade"
        )
    return transition


def _planned_scene_boundaries(plan: Sequence[Any]) -> tuple[SceneBoundaryPoint, ...]:
    """Derive strictly ordered scene-boundary timestamps from planned shots."""

    if not plan:
        raise ValueError("final-film quality gate requires a non-empty shot plan")

    elapsed = 0.0
    boundaries: list[SceneBoundaryPoint] = []
    previous_scene: str | None = None
    previous_shot: Any | None = None

    for shot in plan:
        scene_id = str(getattr(shot, "scene_id", "")).strip()
        duration = float(getattr(shot, "duration", 0.0))
        if not scene_id:
            raise ValueError("planned shot is missing scene_id")
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("planned shot duration must be finite and positive")

        if previous_scene is not None and scene_id != previous_scene:
            boundaries.append(
                SceneBoundaryPoint(
                    from_scene_id=previous_scene,
                    to_scene_id=scene_id,
                    boundary_seconds=elapsed,
                    transition=_planned_transition(previous_shot, shot),
                )
            )

        next_elapsed = elapsed + duration
        if not math.isfinite(next_elapsed):
            raise ValueError("planned shot timeline must remain finite")
        elapsed = next_elapsed
        previous_scene = scene_id
        previous_shot = shot

    return tuple(boundaries)


def _planned_duration_seconds(plan: Sequence[Any]) -> float:
    if not plan:
        raise ValueError("final-film quality gate requires a non-empty shot plan")
    total = 0.0
    for shot in plan:
        duration = float(getattr(shot, "duration", 0.0))
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("planned shot duration must be finite and positive")
        total += duration
        if not math.isfinite(total):
            raise ValueError("planned shot timeline must remain finite")
    return total


@dataclass(slots=True)
class MeasuredFinalFilmGate:
    """Plan-aware measured post-assembly QC used by production FIRST FILM.

    Whole-film temporal sampling detects black/frozen/drifting output. Scene-boundary
    sampling validates authored edit semantics. Duration integrity independently
    proves that the assembled container covers the authored shot plan. Production
    callers can additionally require measured encoded audio; this is opt-in for
    backwards compatibility with intentionally silent legacy/test films.
    """

    temporal_evaluator: FFmpegTemporalFilmEvaluator | None = None
    boundary_evaluator: FFmpegSceneBoundaryEvaluator | None = None
    duration_evaluator: FFprobeDurationIntegrityGate | None = None
    audio_evaluator: FinalFilmAudioIntegrityGate | None = None
    require_audio: bool = False

    def __post_init__(self) -> None:
        if self.temporal_evaluator is None:
            self.temporal_evaluator = FFmpegTemporalFilmEvaluator()
        if self.boundary_evaluator is None:
            self.boundary_evaluator = FFmpegSceneBoundaryEvaluator()
        if self.duration_evaluator is None:
            self.duration_evaluator = FFprobeDurationIntegrityGate()
        if self.require_audio and self.audio_evaluator is None:
            self.audio_evaluator = FinalFilmAudioIntegrityGate()

    def evaluate(
        self, movie_path: str | Path, plan: Sequence[Any]
    ) -> MeasuredFinalFilmReport:
        source = Path(movie_path)
        temporal = self.temporal_evaluator.evaluate(source)
        duration = self.duration_evaluator.evaluate(source, plan)
        boundary_points = _planned_scene_boundaries(plan)
        boundary_report = (
            self.boundary_evaluator.evaluate(source, boundary_points)
            if boundary_points
            else None
        )

        audio_report: AudioIntegrityReport | None = None
        if self.require_audio:
            if self.audio_evaluator is None:
                raise RuntimeError(
                    "required final-film audio evaluator is not configured"
                )
            audio_report = self.audio_evaluator.evaluate(
                source,
                expected_duration_seconds=_planned_duration_seconds(plan),
                required=True,
            )

        decisions = [temporal.decision, duration.decision]
        if boundary_report is not None:
            decisions.append(boundary_report.decision)
        if audio_report is not None:
            decisions.append(audio_report.decision)
        if "reject" in decisions:
            decision = "reject"
        elif "warn" in decisions:
            decision = "warn"
        else:
            decision = "accept"

        directives: list[str] = list(temporal.directives)
        directives.extend(duration.directives)
        if boundary_report is not None:
            for item in boundary_report.boundaries:
                directives.extend(item.directives)
        if audio_report is not None:
            directives.extend(audio_report.directives)

        deduped = tuple(dict.fromkeys(str(item) for item in directives if str(item)))
        return MeasuredFinalFilmReport(
            decision=decision,
            directives=deduped,
            temporal=temporal,
            boundaries=boundary_report,
            duration=duration,
            audio=audio_report,
        )


__all__ = [
    "MeasuredFinalFilmGate",
    "MeasuredFinalFilmReport",
    "_planned_duration_seconds",
    "_planned_scene_boundaries",
]
