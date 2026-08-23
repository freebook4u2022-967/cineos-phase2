"""Persistent failure-recovery state for CINEOS model evolution runs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .checkpoint_gate import CheckpointScore
from .evolution_search import EvolutionConfig
from .training_budget import ResourceUsage


@dataclass(frozen=True, slots=True)
class CandidateProgress:
    candidate_id: str
    status: str = "pending"
    checkpoint_path: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"pending", "running", "completed", "failed"}:
            raise ValueError("unsupported candidate progress status")


@dataclass(slots=True)
class EvolutionResumeState:
    run_id: str
    current_generation: int = 1
    current_config: EvolutionConfig | None = None
    best_score: CheckpointScore | None = None
    usage: ResourceUsage = field(default_factory=ResourceUsage)
    candidates: tuple[CandidateProgress, ...] = ()
    completed_generations: tuple[int, ...] = ()
    schema: str = "cineos-evolution-resume/0.1"

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.current_generation < 1:
            raise ValueError("current_generation must be at least 1")

    def candidate(self, candidate_id: str) -> CandidateProgress | None:
        return next(
            (item for item in self.candidates if item.candidate_id == candidate_id),
            None,
        )

    def with_candidate(self, progress: CandidateProgress) -> EvolutionResumeState:
        items = [item for item in self.candidates if item.candidate_id != progress.candidate_id]
        items.append(progress)
        self.candidates = tuple(items)
        return self

    def next_pending_candidate(self) -> CandidateProgress | None:
        return next((item for item in self.candidates if item.status == "pending"), None)


class EvolutionStateStore:
    """Atomically save and restore the last safe evolution checkpoint."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def _config_payload(config: EvolutionConfig | None):
        return None if config is None else asdict(config)

    @staticmethod
    def _score_payload(score: CheckpointScore | None):
        return None if score is None else asdict(score)

    def save(self, state: EvolutionResumeState) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": state.schema,
            "run_id": state.run_id,
            "current_generation": state.current_generation,
            "current_config": self._config_payload(state.current_config),
            "best_score": self._score_payload(state.best_score),
            "usage": asdict(state.usage),
            "candidates": [asdict(item) for item in state.candidates],
            "completed_generations": list(state.completed_generations),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        return self.path

    def load(self) -> EvolutionResumeState:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema") != "cineos-evolution-resume/0.1":
            raise ValueError("unsupported evolution resume schema")
        current_config = payload.get("current_config")
        best_score = payload.get("best_score")
        return EvolutionResumeState(
            run_id=payload["run_id"],
            current_generation=int(payload["current_generation"]),
            current_config=(
                EvolutionConfig(**current_config) if current_config is not None else None
            ),
            best_score=(
                CheckpointScore(**best_score) if best_score is not None else None
            ),
            usage=ResourceUsage(**payload["usage"]),
            candidates=tuple(
                CandidateProgress(**item) for item in payload.get("candidates", [])
            ),
            completed_generations=tuple(payload.get("completed_generations", [])),
        )

    def exists(self) -> bool:
        return self.path.is_file()


def recover_interrupted_candidates(state: EvolutionResumeState) -> EvolutionResumeState:
    """Reset in-flight candidates to pending so a crashed run can retry safely."""
    recovered = []
    for item in state.candidates:
        if item.status == "running":
            recovered.append(
                CandidateProgress(
                    candidate_id=item.candidate_id,
                    status="pending",
                    checkpoint_path=item.checkpoint_path,
                    error="recovered from interrupted training attempt",
                )
            )
        else:
            recovered.append(item)
    state.candidates = tuple(recovered)
    return state
