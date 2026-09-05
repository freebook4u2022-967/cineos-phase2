"""Checkpoint promotion gate for CINEOS model evolution."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CheckpointScore:
    checkpoint_id: str
    reconstruction_mse: float
    same_character_scene_distance: float
    different_character_same_scene_distance: float
    identity_consistency_score: float

    @property
    def quality_score(self) -> float:
        reconstruction_quality = 1.0 / (1.0 + max(0.0, self.reconstruction_mse))
        conditioning = (
            max(0.0, self.same_character_scene_distance)
            + max(0.0, self.different_character_same_scene_distance)
        ) / 2.0
        return (
            0.40 * reconstruction_quality
            + 0.25 * conditioning
            + 0.35 * max(0.0, min(1.0, self.identity_consistency_score))
        )


@dataclass(frozen=True, slots=True)
class CheckpointPromotionDecision:
    promoted: bool
    candidate_score: float
    incumbent_score: float | None
    reason: str


@dataclass(slots=True)
class CheckpointBenchmarkGate:
    minimum_improvement: float = 0.002

    def evaluate(
        self,
        candidate: CheckpointScore,
        incumbent: CheckpointScore | None,
    ) -> CheckpointPromotionDecision:
        candidate_score = candidate.quality_score
        if incumbent is None:
            return CheckpointPromotionDecision(
                True, candidate_score, None, "first eligible checkpoint"
            )
        incumbent_score = incumbent.quality_score
        improvement = candidate_score - incumbent_score
        promoted = improvement >= self.minimum_improvement
        reason = (
            f"quality improved by {improvement:.6f}"
            if promoted
            else f"quality improvement {improvement:.6f} below promotion threshold"
        )
        return CheckpointPromotionDecision(
            promoted,
            candidate_score,
            incumbent_score,
            reason,
        )

    def promote_if_better(
        self,
        candidate: CheckpointScore,
        incumbent: CheckpointScore | None,
        registry_path: str | Path,
    ) -> CheckpointPromotionDecision:
        decision = self.evaluate(candidate, incumbent)
        if decision.promoted:
            destination = Path(registry_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(
                    {
                        "best_checkpoint": asdict(candidate),
                        "quality_score": candidate.quality_score,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return decision
