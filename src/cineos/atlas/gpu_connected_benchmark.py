"""Fail-closed 5-10 shot GPU benchmark for connected CINEOS film evidence.

The benchmark deliberately sits above the single-shot foundation smoke path. It
only writes a benchmark manifest after every requested shot has produced fresh,
hash-bound video evidence. Third-party pretrained weights remain explicitly
identified by the selected :class:`FoundationExecutionProfile`; the benchmark is
CINEOS orchestration/evidence, not a claim that those weights are CINEOS-native.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .foundation_profiles import FoundationExecutionProfile
from .gpu_foundation_smoke import (
    GPUFoundationExecutionReceipt,
    execute_foundation_gpu_shot,
)
from .native_request import NativeShotRequest


class GPUConnectedBenchmarkError(RuntimeError):
    """Raised when connected-shot benchmark evidence is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class GPUConnectedBenchmarkReceipt:
    """Auditable evidence for one successful connected 5-10 shot GPU run."""

    benchmark_id: str
    profile_id: str
    origin: str
    shot_receipts: tuple[GPUFoundationExecutionReceipt, ...]
    chain_sha256: str
    total_output_bytes: int
    elapsed_seconds: float
    manifest_path: str
    quality_reports: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cineos-gpu-connected-benchmark/0.1",
            "benchmark_id": self.benchmark_id,
            "profile_id": self.profile_id,
            "origin": self.origin,
            "shot_count": len(self.shot_receipts),
            "chain_sha256": self.chain_sha256,
            "total_output_bytes": self.total_output_bytes,
            "elapsed_seconds": self.elapsed_seconds,
            "manifest_path": self.manifest_path,
            "quality_gate_applied": bool(self.quality_reports),
            "quality_reports": list(self.quality_reports),
            "shots": [receipt.to_dict() for receipt in self.shot_receipts],
        }


def _previous_shot_id(request: NativeShotRequest) -> str | None:
    missing = object()
    canonical = request.continuity.get("previous_shot", missing)
    legacy = request.continuity.get("previous_shot_id", missing)

    def normalize(value: Any, *, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise GPUConnectedBenchmarkError(
                f"shot {request.scene_id}/{request.shot_id} {field} must be a string or null"
            )
        value = value.strip()
        return value or None

    if canonical is not missing and legacy is not missing:
        canonical_value = normalize(canonical, field="previous_shot")
        legacy_value = normalize(legacy, field="previous_shot_id")
        if canonical_value != legacy_value:
            raise GPUConnectedBenchmarkError(
                f"shot {request.scene_id}/{request.shot_id} has conflicting "
                "previous_shot and previous_shot_id continuity links"
            )
        return canonical_value

    if canonical is not missing:
        return normalize(canonical, field="previous_shot")
    if legacy is not missing:
        return normalize(legacy, field="previous_shot_id")
    return None


def _validate_continuity_chain(requests: Sequence[NativeShotRequest]) -> None:
    """Require an explicit linear handoff instead of accepting unrelated clips.

    A connected benchmark is film-level evidence only when every shot after the
    first points at the immediately preceding shot. This deliberately fails closed:
    callers must compile/rebuild requests after fixing continuity metadata so the
    request hashes bind the exact chain being rendered.

    ``previous_shot`` is the canonical key. ``previous_shot_id`` remains accepted
    as a compatibility alias for existing competitive-benchmark requests. Supplying
    both is allowed only when they normalize to the same value.
    """
    first = requests[0]
    if _previous_shot_id(first) is not None:
        raise GPUConnectedBenchmarkError(
            f"first benchmark shot {first.scene_id}/{first.shot_id} must not declare previous_shot"
        )

    for previous, current in zip(requests, requests[1:]):
        declared = _previous_shot_id(current)
        if declared != previous.shot_id:
            raise GPUConnectedBenchmarkError(
                "connected GPU benchmark continuity chain is broken: "
                f"{current.scene_id}/{current.shot_id} declares previous_shot={declared!r}, "
                f"expected {previous.shot_id!r}"
            )


def _validate_requests(requests: Sequence[NativeShotRequest]) -> None:
    if not 5 <= len(requests) <= 10:
        raise GPUConnectedBenchmarkError(
            "connected GPU benchmark requires between 5 and 10 shots"
        )

    identities: set[tuple[str, str]] = set()
    for request in requests:
        identity = (request.scene_id, request.shot_id)
        if identity in identities:
            raise GPUConnectedBenchmarkError(
                f"duplicate scene/shot identity in benchmark: {identity!r}"
            )
        identities.add(identity)
        if not request.approved_reference_ids:
            raise GPUConnectedBenchmarkError(
                f"shot {request.scene_id}/{request.shot_id} has no approved identity references"
            )

        supplied_hash = request.content_hash
        expected_hash = request.refresh_hash()
        if not supplied_hash or supplied_hash != expected_hash:
            raise GPUConnectedBenchmarkError(
                f"shot {request.scene_id}/{request.shot_id} request hash is missing or stale"
            )

    _validate_continuity_chain(requests)


def _remove_stale_manifest(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_file():
        raise GPUConnectedBenchmarkError(
            f"benchmark manifest path is not a file: {path}"
        )
    try:
        path.unlink()
    except OSError as exc:
        raise GPUConnectedBenchmarkError(
            f"cannot remove stale benchmark manifest: {path}"
        ) from exc


def _chain_digest(receipts: Sequence[GPUFoundationExecutionReceipt]) -> str:
    digest = hashlib.sha256()
    for receipt in receipts:
        digest.update(receipt.result.request_hash.encode("ascii"))
        digest.update(b"\0")
        digest.update(receipt.output_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_unique_render_evidence(
    receipt: GPUFoundationExecutionReceipt,
    *,
    seen_paths: set[str],
    seen_hashes: set[str],
) -> None:
    """Reject recycled output masquerading as a connected multi-shot render.

    A benchmark is useful only when each shot owns distinct render evidence.
    Reusing one artifact path is always invalid. Reusing an identical payload hash
    across different shot identities is also rejected because it means the run did
    not demonstrate distinct shot execution, even if metadata differs.
    """
    try:
        artifact_path = str(Path(receipt.result.output_path).resolve(strict=False))
    except OSError as exc:
        raise GPUConnectedBenchmarkError(
            f"cannot resolve connected benchmark artifact: {receipt.result.output_path}"
        ) from exc

    if artifact_path in seen_paths:
        raise GPUConnectedBenchmarkError(
            "connected GPU benchmark reused one output artifact across multiple shots"
        )
    if receipt.output_sha256 in seen_hashes:
        raise GPUConnectedBenchmarkError(
            "connected GPU benchmark produced duplicate video payloads across multiple shots"
        )

    seen_paths.add(artifact_path)
    seen_hashes.add(receipt.output_sha256)


def _validated_quality_report(
    report: Mapping[str, Any],
    *,
    request: NativeShotRequest,
) -> dict[str, Any]:
    """Normalize one measured quality decision and fail closed on weak evidence."""
    accepted = report.get("accepted")
    if not isinstance(accepted, bool):
        raise GPUConnectedBenchmarkError(
            f"shot {request.scene_id}/{request.shot_id} quality report is missing boolean accepted"
        )

    score = report.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise GPUConnectedBenchmarkError(
            f"shot {request.scene_id}/{request.shot_id} quality report is missing numeric score"
        )
    numeric_score = float(score)
    if not 0.0 <= numeric_score <= 1.0:
        raise GPUConnectedBenchmarkError(
            f"shot {request.scene_id}/{request.shot_id} quality score must be between 0 and 1"
        )

    normalized = dict(report)
    normalized["scene_id"] = request.scene_id
    normalized["shot_id"] = request.shot_id
    normalized["request_hash"] = request.content_hash
    normalized["score"] = numeric_score
    try:
        json.dumps(normalized, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise GPUConnectedBenchmarkError(
            f"shot {request.scene_id}/{request.shot_id} quality report is not JSON serializable"
        ) from exc

    if not accepted:
        failed_metrics = normalized.get("failed_metrics")
        if isinstance(failed_metrics, list) and failed_metrics:
            reason = ", ".join(str(item) for item in failed_metrics)
        else:
            reason = "quality gate rejected rendered artifact"
        raise GPUConnectedBenchmarkError(
            f"shot {request.scene_id}/{request.shot_id} failed connected quality gate: {reason}"
        )
    return normalized


def run_connected_gpu_benchmark(
    benchmark_id: str,
    requests: Sequence[NativeShotRequest],
    profile: FoundationExecutionProfile,
    *,
    output_dir: str | Path,
    shot_executor: Callable[
        ..., GPUFoundationExecutionReceipt
    ] = execute_foundation_gpu_shot,
    shot_executor_kwargs: dict[str, Any] | None = None,
    quality_evaluator: Callable[..., Mapping[str, Any]] | None = None,
) -> GPUConnectedBenchmarkReceipt:
    """Execute 5-10 connected shots and persist evidence only after full success.

    ``shot_executor`` is injectable for deterministic regression tests. Production
    callers should use the default, which performs CUDA preflight and real
    Diffusers-backed rendering for every shot.

    When ``quality_evaluator`` is supplied, every rendered artifact must return an
    accepted, normalized report before the benchmark can complete. This lets real
    identity/temporal/motion observers turn the connected GPU benchmark into a
    fail-closed film-quality gate without breaking legacy callers that only need
    execution-integrity evidence.
    """
    if not benchmark_id.strip():
        raise GPUConnectedBenchmarkError("benchmark_id must not be empty")
    _validate_requests(requests)
    if quality_evaluator is not None and not callable(quality_evaluator):
        raise TypeError("quality_evaluator must be callable")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / f"{benchmark_id}.gpu-benchmark.json"
    _remove_stale_manifest(manifest)

    receipts: list[GPUFoundationExecutionReceipt] = []
    quality_reports: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    kwargs = dict(shot_executor_kwargs or {})
    started = perf_counter()
    try:
        for request in requests:
            receipt = shot_executor(
                request,
                profile,
                output_dir=output_root,
                **kwargs,
            )
            if (
                receipt.profile_id != profile.profile_id
                or receipt.origin != profile.origin
            ):
                raise GPUConnectedBenchmarkError(
                    "shot receipt provenance does not match selected benchmark profile"
                )
            if receipt.result.request_hash != request.content_hash:
                raise GPUConnectedBenchmarkError(
                    "shot receipt request hash does not match connected benchmark request"
                )
            _validate_unique_render_evidence(
                receipt,
                seen_paths=seen_paths,
                seen_hashes=seen_hashes,
            )
            if quality_evaluator is not None:
                raw_report = quality_evaluator(
                    receipt.result.output_path,
                    shot=request,
                    attempt_index=0,
                )
                if not isinstance(raw_report, Mapping):
                    raise GPUConnectedBenchmarkError(
                        f"shot {request.scene_id}/{request.shot_id} quality evaluator must return a mapping"
                    )
                quality_reports.append(
                    _validated_quality_report(raw_report, request=request)
                )
            receipts.append(receipt)
    except Exception:
        # Partial videos may remain useful for debugging, but no completed manifest
        # is allowed to survive a failed connected benchmark.
        _remove_stale_manifest(manifest)
        raise

    elapsed = perf_counter() - started
    chain_sha256 = _chain_digest(receipts)
    total_output_bytes = sum(receipt.output_bytes for receipt in receipts)
    completed = GPUConnectedBenchmarkReceipt(
        benchmark_id=benchmark_id,
        profile_id=profile.profile_id,
        origin=profile.origin,
        shot_receipts=tuple(receipts),
        chain_sha256=chain_sha256,
        total_output_bytes=total_output_bytes,
        elapsed_seconds=elapsed,
        manifest_path=str(manifest),
        quality_reports=tuple(quality_reports),
    )

    payload = completed.to_dict()
    payload["foundation_profile"] = profile.snapshot()
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
        raise GPUConnectedBenchmarkError(
            f"cannot persist connected GPU benchmark manifest: {manifest}"
        ) from exc

    return completed


__all__ = [
    "GPUConnectedBenchmarkError",
    "GPUConnectedBenchmarkReceipt",
    "run_connected_gpu_benchmark",
]
