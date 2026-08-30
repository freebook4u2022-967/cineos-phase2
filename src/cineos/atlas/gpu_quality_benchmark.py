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
    """Evaluate and, when enabled, correctively rerender each connected shot.

    The same ``NativeShotRequest`` object is promoted in place when a retry is
    required. This is intentional: the outer connected-benchmark runner validates
    the receipt against the request hash *after* the executor returns, so the
    accepted retry becomes the canonical, auditable request revision instead of
    hiding a changed seed/prompt behind the original hash.

    Retries are opt-in through ``max_attempts``. The default remains one attempt
    for backwards compatibility and fail-closed behavior.
    """

    evaluator: QualityEvaluator
    executor: ShotExecutor = execute_foundation_gpu_shot
    max_attempts: int = 1
    retry_seed_stride: int = 7919
    reports: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.retry_seed_stride < 1:
            raise ValueError("retry_seed_stride must be positive")

    @staticmethod
    def _quality_directives(report: dict[str, Any]) -> list[str]:
        raw = report.get("directives") or []
        if not isinstance(raw, (list, tuple)):
            return []
        return [str(item).strip() for item in raw if str(item).strip()]

    def _prepare_retry(
        self,
        request: NativeShotRequest,
        *,
        directives: Sequence[str],
        attempt_index: int,
        initial_request_hash: str,
    ) -> None:
        existing = request.metadata.get("quality_directives", [])
        if not isinstance(existing, list):
            existing = []
        merged = [str(item) for item in existing]
        for directive in directives:
            if directive not in merged:
                merged.append(directive)
        request.metadata["quality_directives"] = merged
        request.metadata["quality_retry"] = {
            "schema": "cineos-quality-retry/0.1",
            "attempt_index": attempt_index,
            "initial_request_hash": initial_request_hash,
            "directives": list(directives),
        }
        request.deterministic_seed += self.retry_seed_stride
        request.refresh_hash()

    def __call__(
        self,
        request: NativeShotRequest,
        profile: FoundationExecutionProfile,
        *,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> GPUFoundationExecutionReceipt:
        initial_request_hash = request.content_hash
        attempt_reports: list[dict[str, Any]] = []

        for attempt_index in range(self.max_attempts):
            receipt = self.executor(
                request,
                profile,
                output_dir=output_dir,
                **kwargs,
            )
            raw_report = self.evaluator(
                receipt.result.output_path,
                shot=request,
                attempt_index=attempt_index,
            )
            if not isinstance(raw_report, dict):
                raise GPUQualityBenchmarkError(
                    "quality evaluator must return a dict report"
                )

            report = dict(raw_report)
            report["scene_id"] = request.scene_id
            report["shot_id"] = request.shot_id
            report["attempt_index"] = attempt_index
            report["request_hash"] = request.content_hash
            report["output_sha256"] = receipt.output_sha256
            attempt_reports.append(dict(report))

            if report.get("accepted") is True:
                report["initial_request_hash"] = initial_request_hash
                report["attempt_count"] = attempt_index + 1
                report["rerendered"] = attempt_index > 0
                report["attempts"] = attempt_reports
                self.reports.append(report)
                return receipt

            failed = report.get("failed_metrics") or ["unknown_quality_failure"]
            directives = self._quality_directives(report)
            if attempt_index + 1 < self.max_attempts:
                self._prepare_retry(
                    request,
                    directives=directives,
                    attempt_index=attempt_index + 1,
                    initial_request_hash=initial_request_hash,
                )
                continue

            failure_text = ", ".join(str(item) for item in failed)
            directive_text = "; ".join(directives)
            suffix = f"; directives: {directive_text}" if directive_text else ""
            raise GPUQualityBenchmarkError(
                f"quality gate rejected {request.scene_id}/{request.shot_id} "
                f"after {self.max_attempts} attempt(s): {failure_text}{suffix}"
            )

        raise GPUQualityBenchmarkError("quality rerender loop exited unexpectedly")


def _with_quality_reports(
    receipt: GPUConnectedBenchmarkReceipt,
    reports: Sequence[dict[str, Any]],
) -> GPUConnectedBenchmarkReceipt:
    """Return one receipt whose evidence tier includes the measured quality gate."""
    if len(reports) != len(receipt.shot_receipts):
        raise GPUQualityBenchmarkError(
            "quality evidence count does not match connected benchmark shot count"
        )
    return GPUConnectedBenchmarkReceipt(
        benchmark_id=receipt.benchmark_id,
        profile_id=receipt.profile_id,
        origin=receipt.origin,
        shot_receipts=receipt.shot_receipts,
        chain_sha256=receipt.chain_sha256,
        total_output_bytes=receipt.total_output_bytes,
        elapsed_seconds=receipt.elapsed_seconds,
        manifest_path=receipt.manifest_path,
        quality_reports=tuple(dict(report) for report in reports),
    )


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

    payload["quality_gate_applied"] = True
    payload["quality_reports"] = list(reports)
    payload["production_gpu_evidence"] = receipt.production_gpu_evidence
    payload["evidence_tier"] = receipt.evidence_tier
    payload["quality_gate"] = {
        "schema": "cineos-gpu-connected-quality-gate/0.2",
        "accepted": True,
        "shot_count": len(reports),
        "rerendered_shot_count": sum(
            1 for report in reports if report.get("rerendered") is True
        ),
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
    max_quality_attempts: int = 1,
    retry_seed_stride: int = 7919,
) -> GPUConnectedBenchmarkReceipt:
    """Render 5-10 connected shots with measured accept/reject/rerender evidence.

    With ``max_quality_attempts=1`` this preserves the historical fail-closed
    behavior. Larger values enable deterministic corrective rerenders: evaluator
    directives are injected into the CINEOS request, the seed is advanced, the
    request hash is refreshed, and only the accepted revision enters the connected
    benchmark receipt. Every attempt remains recorded in the quality evidence.
    """
    if not callable(quality_evaluator):
        raise TypeError("quality_evaluator must be callable")
    if max_quality_attempts < 1:
        raise ValueError("max_quality_attempts must be at least 1")

    gated = QualityGatedShotExecutor(
        evaluator=quality_evaluator,
        executor=shot_executor,
        max_attempts=max_quality_attempts,
        retry_seed_stride=retry_seed_stride,
    )
    base_receipt = run_connected_gpu_benchmark(
        benchmark_id,
        requests,
        profile,
        output_dir=output_dir,
        shot_executor=gated,
        shot_executor_kwargs=shot_executor_kwargs,
    )
    receipt = _with_quality_reports(base_receipt, gated.reports)
    _persist_quality_evidence(receipt, gated.reports)
    return receipt


__all__ = [
    "GPUQualityBenchmarkError",
    "QualityGatedShotExecutor",
    "run_quality_gated_connected_gpu_benchmark",
]
