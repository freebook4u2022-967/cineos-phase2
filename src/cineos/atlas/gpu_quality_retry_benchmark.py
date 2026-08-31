"""Quality-driven connected GPU benchmark with auditable rerender lineage.

This module closes the production loop between measured CINEOS sequence quality
and real foundation-backed GPU execution. A rejected shot is retried only through
``build_quality_retry_request`` so every correction has a fresh hash, deterministic
seed, explicit directives, and parent-request lineage. External foundation
provenance remains unchanged and is never represented as CINEOS-native weights.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from .foundation_profiles import FoundationExecutionProfile
from .gpu_connected_benchmark import (
    GPUConnectedBenchmarkError,
    GPUConnectedBenchmarkReceipt,
    _chain_digest,
    _remove_stale_manifest,
    _validate_requests,
    _validate_unique_render_evidence,
)
from .gpu_foundation_smoke import (
    GPUFoundationExecutionReceipt,
    execute_foundation_gpu_shot,
)
from .native_request import NativeShotRequest
from .quality_retry import QualityRetryPolicy, build_quality_retry_request


class GPUQualityRetryBenchmarkError(GPUConnectedBenchmarkError):
    """Raised when a quality-driven connected rerender run cannot be accepted."""


QualityEvaluator = Callable[..., dict[str, Any]]
ShotExecutor = Callable[..., GPUFoundationExecutionReceipt]


def _evaluate(
    evaluator: QualityEvaluator,
    receipt: GPUFoundationExecutionReceipt,
    request: NativeShotRequest,
    *,
    attempt_index: int,
    original_request_hash: str,
) -> dict[str, Any]:
    raw = evaluator(
        receipt.result.output_path,
        shot=request,
        attempt_index=attempt_index,
    )
    if not isinstance(raw, dict):
        raise GPUQualityRetryBenchmarkError(
            "quality evaluator must return a dict report"
        )
    report = dict(raw)
    report.update(
        {
            "scene_id": request.scene_id,
            "shot_id": request.shot_id,
            "attempt_index": attempt_index,
            "original_request_hash": original_request_hash,
            "effective_request_hash": request.content_hash,
            "seed": request.deterministic_seed,
            "output_sha256": receipt.output_sha256,
        }
    )
    return report


def _render_attempt(
    executor: ShotExecutor,
    request: NativeShotRequest,
    profile: FoundationExecutionProfile,
    *,
    output_dir: Path,
    executor_kwargs: dict[str, Any],
) -> GPUFoundationExecutionReceipt:
    receipt = executor(
        request,
        profile,
        output_dir=output_dir,
        **executor_kwargs,
    )
    if receipt.profile_id != profile.profile_id or receipt.origin != profile.origin:
        raise GPUQualityRetryBenchmarkError(
            "shot receipt provenance does not match selected benchmark profile"
        )
    if receipt.result.request_hash != request.content_hash:
        raise GPUQualityRetryBenchmarkError(
            "shot receipt request hash does not match the effective rerender request"
        )
    return receipt


def run_quality_retry_connected_gpu_benchmark(
    benchmark_id: str,
    requests: Sequence[NativeShotRequest],
    profile: FoundationExecutionProfile,
    *,
    output_dir: str | Path,
    quality_evaluator: QualityEvaluator,
    retry_policy: QualityRetryPolicy | None = None,
    shot_executor: ShotExecutor = execute_foundation_gpu_shot,
    shot_executor_kwargs: dict[str, Any] | None = None,
) -> GPUConnectedBenchmarkReceipt:
    """Render 5-10 connected shots with measured reject/correct/rerender attempts.

    Only the final accepted artifact for each shot enters the connected-film chain.
    Every rejected attempt remains represented in the manifest by its request hash,
    seed, quality report, and output digest. The benchmark fails closed and writes
    no completed manifest if any shot exhausts the retry policy.
    """

    if not benchmark_id.strip():
        raise GPUQualityRetryBenchmarkError("benchmark_id must not be empty")
    if not callable(quality_evaluator):
        raise TypeError("quality_evaluator must be callable")

    _validate_requests(requests)
    policy = retry_policy or QualityRetryPolicy()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / f"{benchmark_id}.gpu-quality-retry.json"
    _remove_stale_manifest(manifest)

    receipts: list[GPUFoundationExecutionReceipt] = []
    shot_evidence: list[dict[str, Any]] = []
    accepted_quality_reports: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    kwargs = dict(shot_executor_kwargs or {})
    started = perf_counter()

    try:
        for original in requests:
            original_hash = original.content_hash
            effective = original
            attempts: list[dict[str, Any]] = []
            accepted_receipt: GPUFoundationExecutionReceipt | None = None
            accepted_report: dict[str, Any] | None = None

            for attempt_index in range(policy.max_attempts):
                receipt = _render_attempt(
                    shot_executor,
                    effective,
                    profile,
                    output_dir=output_root,
                    executor_kwargs=kwargs,
                )
                report = _evaluate(
                    quality_evaluator,
                    receipt,
                    effective,
                    attempt_index=attempt_index,
                    original_request_hash=original_hash,
                )
                attempts.append(report)

                if report.get("accepted") is True:
                    accepted_receipt = receipt
                    accepted_report = report
                    break

                if attempt_index + 1 >= policy.max_attempts:
                    failed = report.get("failed_metrics") or ["unknown_quality_failure"]
                    raise GPUQualityRetryBenchmarkError(
                        f"quality retry exhausted for "
                        f"{original.scene_id}/{original.shot_id}: "
                        + ", ".join(str(item) for item in failed)
                    )

                effective = build_quality_retry_request(
                    effective,
                    report,
                    attempt_index=attempt_index + 1,
                    policy=policy,
                )

            if accepted_receipt is None or accepted_report is None:
                raise GPUQualityRetryBenchmarkError(
                    f"no accepted receipt for {original.scene_id}/{original.shot_id}"
                )

            _validate_unique_render_evidence(
                accepted_receipt,
                seen_paths=seen_paths,
                seen_hashes=seen_hashes,
            )
            receipts.append(accepted_receipt)
            accepted_quality_reports.append(accepted_report)
            shot_evidence.append(
                {
                    "scene_id": original.scene_id,
                    "shot_id": original.shot_id,
                    "original_request_hash": original_hash,
                    "accepted_request_hash": accepted_receipt.result.request_hash,
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                }
            )
    except Exception:
        _remove_stale_manifest(manifest)
        raise

    elapsed = perf_counter() - started
    completed = GPUConnectedBenchmarkReceipt(
        benchmark_id=benchmark_id,
        profile_id=profile.profile_id,
        origin=profile.origin,
        shot_receipts=tuple(receipts),
        chain_sha256=_chain_digest(receipts),
        total_output_bytes=sum(receipt.output_bytes for receipt in receipts),
        elapsed_seconds=elapsed,
        manifest_path=str(manifest),
        quality_reports=tuple(accepted_quality_reports),
    )
    payload = completed.to_dict()
    payload["foundation_profile"] = profile.snapshot()
    payload["quality_retry_gate"] = {
        "schema": "cineos-gpu-quality-retry-gate/0.1",
        "accepted": True,
        "policy": {
            "max_attempts": policy.max_attempts,
            "seed_stride": policy.seed_stride,
        },
        "shot_count": len(shot_evidence),
        "shots": shot_evidence,
    }

    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
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
        raise GPUQualityRetryBenchmarkError(
            f"cannot persist quality-retry benchmark evidence: {manifest}"
        ) from exc

    return completed


__all__ = [
    "GPUQualityRetryBenchmarkError",
    "run_quality_retry_connected_gpu_benchmark",
]
