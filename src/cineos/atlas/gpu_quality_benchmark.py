"""Quality-gated connected GPU benchmark for production film evidence.

This module composes the existing real GPU connected-shot benchmark with the
CINEOS-owned sequence quality evaluator. It deliberately keeps pretrained
foundation provenance unchanged: CINEOS owns the acceptance policy, evidence,
and reject/rerender decision, not the external foundation weights.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .foundation_profiles import FoundationExecutionProfile
from .gpu_connected_benchmark import (
    GPUConnectedBenchmarkError,
    GPUConnectedBenchmarkReceipt,
    run_connected_gpu_benchmark,
)
from .gpu_foundation_smoke import (
    GPUFoundationExecutionReceipt,
    execute_foundation_gpu_shot,
)
from .native_request import NativeShotRequest


class GPUQualityBenchmarkError(GPUConnectedBenchmarkError):
    """Raised when measured render quality rejects a connected GPU shot."""


QualityEvaluator = Callable[..., dict[str, Any]]
ShotExecutor = Callable[..., GPUFoundationExecutionReceipt]


@dataclass(slots=True)
class QualityGatedShotExecutor:
    """Evaluate every freshly rendered artifact before it can enter a sequence."""

    evaluator: QualityEvaluator
    executor: ShotExecutor = execute_foundation_gpu_shot
    reports: list[dict[str, Any]] = field(default_factory=list)

    def __call__(
        self,
        request: NativeShotRequest,
        profile: FoundationExecutionProfile,
        *,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> GPUFoundationExecutionReceipt:
        receipt = self.executor(
            request,
            profile,
            output_dir=output_dir,
            **kwargs,
        )
        raw_report = self.evaluator(
            receipt.result.output_path,
            shot=request,
            attempt_index=0,
        )
        if not isinstance(raw_report, dict):
            raise GPUQualityBenchmarkError(
                "quality evaluator must return a dict report"
            )

        report = dict(raw_report)
        report["scene_id"] = request.scene_id
        report["shot_id"] = request.shot_id
        report["request_hash"] = request.content_hash
        report["output_sha256"] = receipt.output_sha256
        self.reports.append(report)

        if report.get("accepted") is not True:
            failed = report.get("failed_metrics") or ["unknown_quality_failure"]
            directives = report.get("directives") or []
            failure_text = ", ".join(str(item) for item in failed)
            directive_text = "; ".join(str(item) for item in directives)
            suffix = f"; directives: {directive_text}" if directive_text else ""
            raise GPUQualityBenchmarkError(
                f"quality gate rejected {request.scene_id}/{request.shot_id}: "
                f"{failure_text}{suffix}"
            )

        return receipt


def _persist_quality_evidence(
    receipt: GPUConnectedBenchmarkReceipt,
    reports: Sequence[dict[str, Any]],
) -> None:
    if len(reports) != len(receipt.shot_receipts):
        raise GPUQualityBenchmarkError(
            "quality evidence count does not match connected benchmark shot count"
        )

    manifest = Path(receipt.manifest_path)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GPUQualityBenchmarkError(
            f"cannot read connected benchmark manifest for quality evidence: {manifest}"
        ) from exc

    payload["quality_gate"] = {
        "schema": "cineos-gpu-connected-quality-gate/0.1",
        "accepted": True,
        "shot_count": len(reports),
        "reports": list(reports),
    }
    temporary = manifest.with_suffix(manifest.suffix + ".quality.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise GPUQualityBenchmarkError(
            f"cannot persist quality-gated benchmark evidence: {manifest}"
        ) from exc


def run_quality_gated_connected_gpu_benchmark(
    benchmark_id: str,
    requests: Sequence[NativeShotRequest],
    profile: FoundationExecutionProfile,
    *,
    output_dir: str | Path,
    quality_evaluator: QualityEvaluator,
    shot_executor: ShotExecutor = execute_foundation_gpu_shot,
    shot_executor_kwargs: dict[str, Any] | None = None,
) -> GPUConnectedBenchmarkReceipt:
    """Render 5-10 connected shots and accept only measured passing artifacts.

    A rejected shot aborts the connected benchmark through the existing fail-closed
    path, so no completed benchmark manifest survives. A fully accepted run stores
    hash-bound per-shot quality evidence in the same benchmark manifest.
    """
    if not callable(quality_evaluator):
        raise TypeError("quality_evaluator must be callable")

    gated = QualityGatedShotExecutor(
        evaluator=quality_evaluator,
        executor=shot_executor,
    )
    receipt = run_connected_gpu_benchmark(
        benchmark_id,
        requests,
        profile,
        output_dir=output_dir,
        shot_executor=gated,
        shot_executor_kwargs=shot_executor_kwargs,
    )
    _persist_quality_evidence(receipt, gated.reports)
    return receipt


__all__ = [
    "GPUQualityBenchmarkError",
    "QualityGatedShotExecutor",
    "run_quality_gated_connected_gpu_benchmark",
]
