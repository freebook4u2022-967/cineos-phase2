"""Measured QC-driven rerender orchestration for native video shots.

This module provides the execution primitive used when a real render completes but
fails an explicit visual evaluator. Retries are deterministic, auditable variants
of the original native shot request: each attempt receives a derived seed, records
the original request hash, and must independently pass measured QC before it can be
accepted. Artifact existence alone never upgrades a failed shot to accepted.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cineos.atlas.native_request import NativeShotRequest

from .competitive_benchmark import VideoRenderer, VisualEvaluator


@dataclass(frozen=True, slots=True)
class QualityRetryPolicy:
    """Fail-closed retry limits for measured visual QC."""

    max_attempts: int = 3
    seed_stride: int = 100_003

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.seed_stride < 1:
            raise ValueError("seed_stride must be >= 1")


@dataclass(frozen=True, slots=True)
class QualityRetryAttempt:
    """Auditable evidence for one render/evaluate attempt."""

    attempt: int
    deterministic_seed: int
    request_hash: str
    output_path: str | None
    artifact_bytes: int
    execution_passed: bool
    quality_evaluated: bool
    quality_passed: bool | None
    quality_metrics: dict[str, float]
    notes: tuple[str, ...]
    frame_count: int | None = None

    @property
    def metric_mean(self) -> float | None:
        if not self.quality_metrics:
            return None
        return sum(self.quality_metrics.values()) / len(self.quality_metrics)


@dataclass(frozen=True, slots=True)
class QualityRetryResult:
    """Final QC decision plus the complete retry trail."""

    accepted: bool
    selected_attempt: int | None
    selected_output_path: str | None
    attempts: tuple[QualityRetryAttempt, ...]

    @property
    def rerendered(self) -> bool:
        return len(self.attempts) > 1

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


def _derived_request(
    request: NativeShotRequest,
    *,
    attempt: int,
    seed_stride: int,
) -> NativeShotRequest:
    candidate = copy.deepcopy(request)
    base_hash = request.content_hash or request.refresh_hash()
    candidate.deterministic_seed = (
        request.deterministic_seed + (attempt - 1) * seed_stride
    )
    candidate.metadata = {
        **dict(candidate.metadata),
        "qc_retry": {
            "attempt": attempt,
            "base_request_hash": base_hash,
        },
    }
    candidate.refresh_hash()
    return candidate


def _best_attempt(attempts: list[QualityRetryAttempt]) -> QualityRetryAttempt | None:
    measured = [item for item in attempts if item.metric_mean is not None]
    if measured:
        return max(measured, key=lambda item: float(item.metric_mean or 0.0))
    executed = [item for item in attempts if item.execution_passed]
    if executed:
        return executed[-1]
    return None


def render_with_quality_retries(
    renderer: VideoRenderer,
    evaluator: VisualEvaluator,
    request: NativeShotRequest,
    *,
    policy: QualityRetryPolicy | None = None,
) -> QualityRetryResult:
    """Render until measured QC passes or the retry budget is exhausted.

    Every retry varies only the deterministic seed and retry metadata. Identity,
    scene, camera, continuity, wardrobe, props, and other conditioning stay intact.
    This gives a renderer a new stochastic sample without silently weakening the
    director's constraints. A shot is accepted only when the supplied evaluator
    explicitly returns ``passed=True``.
    """

    active = policy or QualityRetryPolicy()
    attempts: list[QualityRetryAttempt] = []

    for attempt_number in range(1, active.max_attempts + 1):
        candidate = _derived_request(
            request,
            attempt=attempt_number,
            seed_stride=active.seed_stride,
        )
        notes: list[str] = []
        output_path: Path | None = None
        artifact_bytes = 0
        frame_count: int | None = None
        execution_passed = False
        quality_evaluated = False
        quality_passed: bool | None = None
        quality_metrics: dict[str, float] = {}

        try:
            rendered: Any = renderer.render(candidate)
            raw_path = getattr(rendered, "output_path", rendered)
            output_path = Path(raw_path)
            frame_count_value = getattr(rendered, "frame_count", None)
            if frame_count_value is not None:
                frame_count = int(frame_count_value)
            if output_path.is_file():
                artifact_bytes = output_path.stat().st_size
                execution_passed = artifact_bytes > 0
            if not execution_passed:
                notes.append("renderer did not produce a non-empty artifact")
        except Exception as exc:
            notes.append(f"render failed: {type(exc).__name__}: {exc}")

        if execution_passed and output_path is not None:
            try:
                evaluation = evaluator(output_path, candidate)
                quality_evaluated = True
                quality_passed = bool(evaluation.passed)
                quality_metrics = {
                    name: float(value) for name, value in evaluation.metrics.items()
                }
                notes.extend(evaluation.notes)
            except Exception as exc:
                notes.append(f"visual evaluation failed: {type(exc).__name__}: {exc}")

        evidence = QualityRetryAttempt(
            attempt=attempt_number,
            deterministic_seed=candidate.deterministic_seed,
            request_hash=candidate.content_hash,
            output_path=str(output_path) if output_path is not None else None,
            artifact_bytes=artifact_bytes,
            execution_passed=execution_passed,
            quality_evaluated=quality_evaluated,
            quality_passed=quality_passed,
            quality_metrics=quality_metrics,
            notes=tuple(notes),
            frame_count=frame_count,
        )
        attempts.append(evidence)

        if quality_passed:
            return QualityRetryResult(
                accepted=True,
                selected_attempt=attempt_number,
                selected_output_path=evidence.output_path,
                attempts=tuple(attempts),
            )

    selected = _best_attempt(attempts)
    return QualityRetryResult(
        accepted=False,
        selected_attempt=selected.attempt if selected is not None else None,
        selected_output_path=selected.output_path if selected is not None else None,
        attempts=tuple(attempts),
    )


__all__ = [
    "QualityRetryAttempt",
    "QualityRetryPolicy",
    "QualityRetryResult",
    "render_with_quality_retries",
]
