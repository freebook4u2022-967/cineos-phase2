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

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .boundary_eval import FFmpegSceneBoundaryEvaluator, SceneBoundaryPoint
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


def _payload(item: Any) -> Mapping[str, Any]:
    value = getattr(item, "payload", None)
    return value if isinstance(value, Mapping) else {}


def _transition_for_boundary(outgoing: Any, incoming: Any) -> str:
    """Resolve an authored edit contract without guessing from rendered pixels."""
    incoming_payload = _payload(incoming)
    outgoing_payload = _payload(outgoing)
    raw = incoming_payload.get(
        "scene_transition",
        incoming_payload.get(
            "transition",
            outgoing_payload.get(
                "scene_transition", outgoing_payload.get("transition")
            ),
        ),
    )
    if raw is None:
        if bool(incoming_payload.get("continuity_reset", False)) or bool(
            incoming_payload.get("hard_cut", False)
        ):
            return "cut"
        return "match"
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
            f"unsupported scene transition {raw!r}; expected cut, match, or fade"
        )
    return transition


def plan_scene_boundaries(plan: Sequence[Any]) -> tuple[SceneBoundaryPoint, ...]:
    """Convert a shot timeline into measured scene-boundary sample points.

    A boundary is emitted only when the authored ``scene_id`` changes. The edit
    timestamp is the cumulative duration of all shots before the incoming scene.
    Durations must be positive so boundary timestamps are deterministic and safe
    for decoder sampling.
    """
    if not plan:
        raise ValueError("final-film quality gate requires at least one planned shot")

    elapsed = 0.0
    boundaries: list[SceneBoundaryPoint] = []
    previous: Any | None = None
    for item in plan:
        scene_id = str(getattr(item, "scene_id", "")).strip()
        if not scene_id:
            raise ValueError("planned shots require non-empty scene_id values")
        duration = float(getattr(item, "duration", 0.0))
        if duration <= 0.0:
            raise ValueError("planned shot durations must be positive")

        if previous is not None:
            previous_scene_id = str(getattr(previous, "scene_id", "")).strip()
            if scene_id != previous_scene_id:
                boundaries.append(
                    SceneBoundaryPoint(
                        from_scene_id=previous_scene_id,
                        to_scene_id=scene_id,
                        boundary_seconds=elapsed,
                        transition=_transition_for_boundary(previous, item),
                    )
                )
        elapsed += duration
        previous = item

    return tuple(boundaries)


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
