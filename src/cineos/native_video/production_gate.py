"""Production final-film quality gate backed by measured decoded pixels.

The shot-level runtime already rejects bad individual renders. This module closes
an orthogonal production gap: after assembly, it evaluates the *actual movie*
rather than inferring final quality from per-shot acceptance. It combines global
temporal evidence with edit-aware scene-boundary evidence while keeping the film
orchestrator renderer-neutral.

FFmpeg remains only a decoder/sampler through the existing evaluators; it never
generates or repairs visual content. Missing evidence fails closed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .boundary_eval import FFmpegSceneBoundaryEvaluator, SceneBoundaryPoint
from .edit_contract import planned_scene_boundaries as plan_scene_boundaries
from .final_eval import (
    FFmpegTemporalFilmEvaluator,
    SceneBoundaryEvalReport,
    TemporalFilmEvalReport,
)


class TemporalFilmEvaluator(Protocol):
    def evaluate(self, movie_path: str | Path) -> TemporalFilmEvalReport: ...


class SceneBoundaryEvaluator(Protocol):
    def evaluate(
        self,
        movie_path: str | Path,
        boundaries: Sequence[SceneBoundaryPoint],
    ) -> SceneBoundaryEvalReport: ...


@dataclass(frozen=True, slots=True)
class FinalFilmQualityReport:
    """Auditable aggregate decision for a fully assembled movie."""

    decision: str
    temporal: TemporalFilmEvalReport
    scene_boundaries: SceneBoundaryEvalReport | None
    directives: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.decision in {"accept", "warn"}

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe evidence suitable for FilmBuild metadata/checkpoints."""
        return asdict(self)


@dataclass(slots=True)
class MeasuredFinalFilmGate:
    """Fail-closed post-assembly gate using real decoded movie evidence."""

    temporal_evaluator: TemporalFilmEvaluator | None = None
    boundary_evaluator: SceneBoundaryEvaluator | None = None

    def __post_init__(self) -> None:
        if self.temporal_evaluator is None:
            self.temporal_evaluator = FFmpegTemporalFilmEvaluator()
        if self.boundary_evaluator is None:
            self.boundary_evaluator = FFmpegSceneBoundaryEvaluator()

    def evaluate(
        self,
        movie_path: str | Path,
        plan: Sequence[Any],
    ) -> FinalFilmQualityReport:
        """Measure the assembled movie against temporal and edit-plan contracts."""
        source = Path(movie_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        if self.temporal_evaluator is None or self.boundary_evaluator is None:
            raise RuntimeError("measured final-film evaluators are not configured")

        temporal = self.temporal_evaluator.evaluate(source)
        boundaries = plan_scene_boundaries(plan)
        boundary_report = (
            self.boundary_evaluator.evaluate(source, boundaries) if boundaries else None
        )

        directives = list(temporal.directives)
        if boundary_report is not None:
            for boundary in boundary_report.boundaries:
                directives.extend(boundary.directives)

        if temporal.decision == "reject" or (
            boundary_report is not None and boundary_report.decision == "reject"
        ):
            decision = "reject"
        elif temporal.decision == "warn" or (
            boundary_report is not None and boundary_report.decision == "warn"
        ):
            decision = "warn"
        else:
            decision = "accept"

        # Preserve order for forensic readability while deduplicating repeated
        # directives caused by multiple failed boundaries of the same class.
        unique_directives = tuple(dict.fromkeys(directives))
        return FinalFilmQualityReport(
            decision=decision,
            temporal=temporal,
            scene_boundaries=boundary_report,
            directives=unique_directives,
        )
