"""Production final-film quality gate for the CINEOS FIRST FILM path.

This module adapts native video evaluators to the provider-neutral ``FirstFilmRunner``
contract. It combines whole-film temporal evidence, plan-aware scene-boundary
evidence, and encoded-duration integrity. It fails closed when measured quality or
assembly completeness is rejected. External tools are inspectors/decoders only;
no external visual generator participates in acceptance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .boundary_eval import FFmpegSceneBoundaryEvaluator, SceneBoundaryPoint
from .duration_gate import DurationIntegrityReport, FFprobeDurationIntegrityGate
from .final_eval import (
    FFmpegTemporalFilmEvaluator,
    SceneBoundaryEvalReport,
    TemporalFilmEvalReport,
)


@dataclass(frozen=True, slots=True)
class MeasuredFinalFilmReport:
    """Auditable aggregate of final-film pixel and assembly evidence."""

    decision: str
    directives: tuple[str, ...]
    temporal: TemporalFilmEvalReport
    boundaries: SceneBoundaryEvalReport | None = None
    duration: DurationIntegrityReport | None = None

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
        }


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
        if duration <= 0.0:
            raise ValueError("planned shot duration must be positive")

        if previous_scene is not None and scene_id != previous_scene:
            payload = getattr(shot, "payload", {}) or {}
            previous_payload = getattr(previous_shot, "payload", {}) or {}
            transition = (
                str(
                    payload.get(
                        "transition_in",
                        payload.get(
                            "transition", previous_payload.get("transition_out", "cut")
                        ),
                    )
                )
                .strip()
                .lower()
            )
            if transition not in {"cut", "match", "fade"}:
                raise ValueError(
                    f"unsupported planned scene transition {transition!r}; "
                    "expected cut, match, or fade"
                )
            boundaries.append(
                SceneBoundaryPoint(
                    from_scene_id=previous_scene,
                    to_scene_id=scene_id,
                    boundary_seconds=elapsed,
                    transition=transition,
                )
            )

        elapsed += duration
        previous_scene = scene_id
        previous_shot = shot

    return tuple(boundaries)


@dataclass(slots=True)
class MeasuredFinalFilmGate:
    """Plan-aware measured post-assembly QC used by production FIRST FILM.

    Whole-film temporal sampling detects black/frozen/drifting output. Scene-boundary
    sampling validates authored edit semantics. Duration integrity independently
    proves that the assembled container covers the authored shot plan, catching
    truncation or duplicated footage that visual sampling alone can miss.
    """

    temporal_evaluator: FFmpegTemporalFilmEvaluator | None = None
    boundary_evaluator: FFmpegSceneBoundaryEvaluator | None = None
    duration_evaluator: FFprobeDurationIntegrityGate | None = None

    def __post_init__(self) -> None:
        if self.temporal_evaluator is None:
            self.temporal_evaluator = FFmpegTemporalFilmEvaluator()
        if self.boundary_evaluator is None:
            self.boundary_evaluator = FFmpegSceneBoundaryEvaluator()
        if self.duration_evaluator is None:
            self.duration_evaluator = FFprobeDurationIntegrityGate()

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

        decisions = [temporal.decision, duration.decision]
        if boundary_report is not None:
            decisions.append(boundary_report.decision)
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

        deduped = tuple(dict.fromkeys(str(item) for item in directives if str(item)))
        return MeasuredFinalFilmReport(
            decision=decision,
            directives=deduped,
            temporal=temporal,
            boundaries=boundary_report,
            duration=duration,
        )


__all__ = [
    "MeasuredFinalFilmGate",
    "MeasuredFinalFilmReport",
    "_planned_scene_boundaries",
]
