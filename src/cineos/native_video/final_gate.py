"""Production final-film quality gate for the CINEOS FIRST FILM path.

This module adapts native video evaluators to the provider-neutral ``FirstFilmRunner``
contract. It combines whole-film temporal evidence, plan-aware scene-boundary
evidence, encoded-duration integrity, optional measured audio integrity, and a
cryptographic identity of the exact accepted movie artifact. It fails closed when
measured quality or assembly completeness is rejected. External tools are
inspectors/decoders only; no external visual generator participates in acceptance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifact_integrity import NativeArtifactProvenance, provenance_for
from .audio_integrity import AudioIntegrityReport, FinalFilmAudioIntegrityGate
from .boundary_eval import FFmpegSceneBoundaryEvaluator
from .duration_gate import DurationIntegrityReport, FFprobeDurationIntegrityGate
from .edit_contract import (
    planned_duration_seconds as _planned_duration_seconds,
)
from .edit_contract import (
    planned_scene_boundaries as _planned_scene_boundaries,
)
from .final_eval import (
    FFmpegTemporalFilmEvaluator,
    SceneBoundaryEvalReport,
    TemporalFilmEvalReport,
)
from .final_repair import FinalFilmRepairPlan, build_final_film_repair_plan


@dataclass(frozen=True, slots=True)
class MeasuredFinalFilmReport:
    """Auditable aggregate of final-film picture, assembly, audio, and identity."""

    decision: str
    directives: tuple[str, ...]
    temporal: TemporalFilmEvalReport
    artifact: NativeArtifactProvenance
    boundaries: SceneBoundaryEvalReport | None = None
    duration: DurationIntegrityReport | None = None
    audio: AudioIntegrityReport | None = None
    repair_plan: FinalFilmRepairPlan | None = None

    @property
    def accepted(self) -> bool:
        return self.decision in {"accept", "warn"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "directives": list(self.directives),
            "artifact": asdict(self.artifact),
            "temporal": asdict(self.temporal),
            "boundaries": (
                asdict(self.boundaries) if self.boundaries is not None else None
            ),
            "duration": asdict(self.duration) if self.duration is not None else None,
            "audio": asdict(self.audio) if self.audio is not None else None,
            "repair_plan": (
                self.repair_plan.as_dict() if self.repair_plan is not None else None
            ),
        }


@dataclass(slots=True)
class MeasuredFinalFilmGate:
    """Plan-aware measured post-assembly QC used by production FIRST FILM.

    Whole-film temporal sampling detects black/frozen/drifting output. Scene-boundary
    sampling validates authored edit semantics. Duration integrity independently
    proves that the assembled container covers the authored shot plan. Production
    callers can additionally require measured encoded audio; this is opt-in for
    backwards compatibility with intentionally silent legacy/test films.

    Every evaluation first computes SHA-256 provenance for the exact assembled movie.
    This binds the resulting QC evidence to immutable artifact bytes and fails closed
    for missing or empty output before any downstream evaluator is trusted.

    Rejected evidence is also converted into a deterministic remediation plan. The
    plan distinguishes visual timeline, scene continuity, assembly, and audio faults
    so autonomous recovery can regenerate only the smallest safe scope instead of
    blindly discarding healthy film assets.
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
        artifact = provenance_for(source)
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
        repair_plan = build_final_film_repair_plan(
            plan=tuple(plan),
            temporal=temporal,
            boundaries=boundary_report,
            duration=duration,
            audio=audio_report,
        )
        return MeasuredFinalFilmReport(
            decision=decision,
            directives=deduped,
            temporal=temporal,
            artifact=artifact,
            boundaries=boundary_report,
            duration=duration,
            audio=audio_report,
            repair_plan=repair_plan,
        )


__all__ = [
    "MeasuredFinalFilmGate",
    "MeasuredFinalFilmReport",
    "_planned_duration_seconds",
    "_planned_scene_boundaries",
]
