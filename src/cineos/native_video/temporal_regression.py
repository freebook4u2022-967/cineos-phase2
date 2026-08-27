"""Version-to-version temporal quality regression gates for CINEOS native video.

Production upgrades must not silently lower measured film quality. This module
compares deterministic final-film temporal and scene-boundary evidence for the
same benchmark workload. It is intentionally dependency-free so release CI can
fail closed before a new native model/runtime generation is promoted.
"""

from __future__ import annotations

from dataclasses import dataclass

from .final_eval import SceneBoundaryEvalReport, TemporalFilmEvalReport


@dataclass(frozen=True, slots=True)
class TemporalRegressionSnapshot:
    """Normalized measured evidence for one repeatable film benchmark."""

    benchmark_id: str
    frame_count: int
    black_frame_ratio: float
    frozen_transition_ratio: float
    hard_cut_transition_ratio: float
    mean_interframe_mad: float
    scene_boundary_reject_count: int
    scene_boundary_warn_count: int
    mean_boundary_mad: float

    def __post_init__(self) -> None:
        if not self.benchmark_id:
            raise ValueError("benchmark_id cannot be empty")
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")
        ratios = (
            self.black_frame_ratio,
            self.frozen_transition_ratio,
            self.hard_cut_transition_ratio,
        )
        if any(not 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("temporal ratios must be in [0, 1]")
        if min(self.mean_interframe_mad, self.mean_boundary_mad) < 0.0:
            raise ValueError("MAD metrics must be non-negative")
        if min(self.scene_boundary_reject_count, self.scene_boundary_warn_count) < 0:
            raise ValueError("scene-boundary counts must be non-negative")

    @classmethod
    def from_reports(
        cls,
        benchmark_id: str,
        temporal: TemporalFilmEvalReport,
        boundaries: SceneBoundaryEvalReport,
    ) -> "TemporalRegressionSnapshot":
        """Build a snapshot only from measured evaluator reports."""

        return cls(
            benchmark_id=benchmark_id,
            frame_count=temporal.frame_count,
            black_frame_ratio=temporal.black_frame_ratio,
            frozen_transition_ratio=temporal.frozen_transition_ratio,
            hard_cut_transition_ratio=temporal.hard_cut_transition_ratio,
            mean_interframe_mad=temporal.mean_interframe_mad,
            scene_boundary_reject_count=boundaries.reject_count,
            scene_boundary_warn_count=boundaries.warn_count,
            mean_boundary_mad=boundaries.mean_boundary_mad,
        )


@dataclass(frozen=True, slots=True)
class TemporalRegressionPolicy:
    """Allowed quality movement between baseline and candidate generations."""

    max_black_ratio_increase: float = 0.005
    max_frozen_ratio_increase: float = 0.02
    max_hard_cut_ratio_increase: float = 0.10
    max_boundary_reject_increase: int = 0
    max_boundary_warn_increase: int = 0
    min_motion_retention: float = 0.70
    require_same_frame_count: bool = True

    def __post_init__(self) -> None:
        deltas = (
            self.max_black_ratio_increase,
            self.max_frozen_ratio_increase,
            self.max_hard_cut_ratio_increase,
        )
        if any(not 0.0 <= value <= 1.0 for value in deltas):
            raise ValueError("ratio regression tolerances must be in [0, 1]")
        if min(self.max_boundary_reject_increase, self.max_boundary_warn_increase) < 0:
            raise ValueError("boundary regression tolerances must be non-negative")
        if not 0.0 <= self.min_motion_retention <= 1.0:
            raise ValueError("min_motion_retention must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TemporalRegressionReport:
    """Auditable result of comparing a candidate against a trusted baseline."""

    benchmark_id: str
    decision: str
    directives: tuple[str, ...]
    black_ratio_delta: float
    frozen_ratio_delta: float
    hard_cut_ratio_delta: float
    boundary_reject_delta: int
    boundary_warn_delta: int
    motion_retention: float

    @property
    def accepted(self) -> bool:
        return self.decision == "accept"


def compare_temporal_regression(
    baseline: TemporalRegressionSnapshot,
    candidate: TemporalRegressionSnapshot,
    policy: TemporalRegressionPolicy | None = None,
) -> TemporalRegressionReport:
    """Reject a candidate release when repeatable film evidence regresses.

    The benchmark identity must match. Frame-count equality is required by
    default because changing the sampled workload can hide frozen/black regions;
    callers may explicitly relax that rule for a versioned benchmark migration.
    """

    if baseline.benchmark_id != candidate.benchmark_id:
        raise ValueError("baseline and candidate benchmark_id must match")

    active_policy = policy or TemporalRegressionPolicy()
    directives: list[str] = []

    if active_policy.require_same_frame_count and baseline.frame_count != candidate.frame_count:
        directives.append(
            "candidate temporal benchmark frame count differs from trusted baseline"
        )

    black_delta = candidate.black_frame_ratio - baseline.black_frame_ratio
    frozen_delta = candidate.frozen_transition_ratio - baseline.frozen_transition_ratio
    hard_cut_delta = candidate.hard_cut_transition_ratio - baseline.hard_cut_transition_ratio
    reject_delta = (
        candidate.scene_boundary_reject_count - baseline.scene_boundary_reject_count
    )
    warn_delta = candidate.scene_boundary_warn_count - baseline.scene_boundary_warn_count

    if black_delta > active_policy.max_black_ratio_increase:
        directives.append("black-frame ratio regressed beyond release tolerance")
    if frozen_delta > active_policy.max_frozen_ratio_increase:
        directives.append("frozen-transition ratio regressed beyond release tolerance")
    if hard_cut_delta > active_policy.max_hard_cut_ratio_increase:
        directives.append("hard-cut ratio regressed beyond release tolerance")
    if reject_delta > active_policy.max_boundary_reject_increase:
        directives.append("scene-boundary reject count increased")
    if warn_delta > active_policy.max_boundary_warn_increase:
        directives.append("scene-boundary warning count increased")

    if baseline.mean_interframe_mad <= 0.0:
        motion_retention = 1.0 if candidate.mean_interframe_mad >= 0.0 else 0.0
    else:
        motion_retention = candidate.mean_interframe_mad / baseline.mean_interframe_mad
    if motion_retention < active_policy.min_motion_retention:
        directives.append("measured temporal motion collapsed versus trusted baseline")

    decision = "reject" if directives else "accept"
    return TemporalRegressionReport(
        benchmark_id=baseline.benchmark_id,
        decision=decision,
        directives=tuple(directives),
        black_ratio_delta=black_delta,
        frozen_ratio_delta=frozen_delta,
        hard_cut_ratio_delta=hard_cut_delta,
        boundary_reject_delta=reject_delta,
        boundary_warn_delta=warn_delta,
        motion_retention=motion_retention,
    )
