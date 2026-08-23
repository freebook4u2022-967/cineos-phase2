"""Automated checkpoint tournament for CINEOS model evolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .checkpoint_gate import CheckpointBenchmarkGate, CheckpointScore


@dataclass(frozen=True, slots=True)
class TournamentCandidate:
    candidate_id: str
    checkpoint_path: str
    score: CheckpointScore


@dataclass(frozen=True, slots=True)
class TournamentResult:
    ranked: tuple[TournamentCandidate, ...]
    winner: TournamentCandidate
    promoted: bool
    reason: str


CandidateTrainer = Callable[[str, Path], TournamentCandidate]


@dataclass(slots=True)
class AutomatedModelTournament:
    gate: CheckpointBenchmarkGate

    def run(
        self,
        candidate_ids: tuple[str, ...],
        trainer: CandidateTrainer,
        work_dir: str | Path,
        *,
        incumbent: CheckpointScore | None = None,
    ) -> TournamentResult:
        if not candidate_ids:
            raise ValueError("model tournament requires at least one candidate")
        destination = Path(work_dir)
        destination.mkdir(parents=True, exist_ok=True)
        candidates = tuple(
            trainer(candidate_id, destination / candidate_id)
            for candidate_id in candidate_ids
        )
        ranked = tuple(
            sorted(candidates, key=lambda item: item.score.quality_score, reverse=True)
        )
        winner = ranked[0]
        decision = self.gate.promote_if_better(
            winner.score,
            incumbent,
            destination / "best_checkpoint.json",
        )
        return TournamentResult(
            ranked=ranked,
            winner=winner,
            promoted=decision.promoted,
            reason=decision.reason,
        )
