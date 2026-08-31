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


def _validated_quality_report(
    report: Mapping[str, Any],
    *,
    request: Any,
    receipt: Any | None = None,
) -> dict[str, Any]:
    """Normalize one quality report and fail closed on substituted render evidence.

    Generic regression reports remain backward compatible. When an evaluator marks
    a report as production measurement evidence, however, its measurement digest
    must match the exact rendered receipt. This prevents stale or substituted QC
    measurements from being attached to a different accepted artifact.
    """
    if not isinstance(report, Mapping):
        raise GPUConnectedBenchmarkError("quality report must be a mapping")

    normalized = dict(report)
    normalized.setdefault("scene_id", getattr(request, "scene_id", None))
    normalized.setdefault("shot_id", getattr(request, "shot_id", None))
    normalized.setdefault(
        "effective_request_hash", getattr(request, "content_hash", None)
    )

    if normalized.get("production_measurement_evidence") is not True:
        return normalized

    measurement = normalized.get("measurement")
    if not isinstance(measurement, Mapping):
        raise GPUConnectedBenchmarkError(
            "production quality report requires artifact-bound measurement evidence"
        )
    if measurement.get("schema") != "cineos-sequence-quality-measurement/0.1":
        raise GPUConnectedBenchmarkError(
            "production quality report has unsupported measurement evidence schema"
        )
    observer_id = measurement.get("observer_id")
    if not isinstance(observer_id, str) or not observer_id.strip():
        raise GPUConnectedBenchmarkError(
            "production quality report requires a measurement observer id"
        )

    artifact_sha256 = measurement.get("artifact_sha256")
    receipt_sha256 = getattr(receipt, "output_sha256", None)
    if (
        not isinstance(artifact_sha256, str)
        or len(artifact_sha256) != 64
        or not isinstance(receipt_sha256, str)
        or artifact_sha256 != receipt_sha256
    ):
        raise GPUConnectedBenchmarkError(
            "production quality measurement does not match rendered artifact hash"
        )

    normalized["measurement"] = dict(measurement)
    return normalized


def _production_gpu_evidence(
    receipts: Sequence[GPUFoundationExecutionReceipt],
) -> bool:
    """Return true only for receipts from the unmodified default GPU runtime.

    Older receipts and deterministic test executors do not carry runtime
    provenance and therefore fail closed. This intentionally separates a useful
    orchestration regression benchmark from evidence that CINEOS actually ran its
    default CUDA + Diffusers execution path.
    """
    if not receipts:
        return False
    for receipt in receipts:
        provenance = receipt.runtime_provenance
        if not isinstance(provenance, Mapping):
            return False
        if provenance.get("schema") != "cineos-gpu-runtime-provenance/0.1":
            return False
        if provenance.get("runtime_mode") != "default":
            return False
        if provenance.get("production_default_runtime") is not True:
            return False
        device = provenance.get("cuda_device")
        if not isinstance(device, str) or not device.startswith("cuda"):
            return False
    return True


def _production_quality_evidence(
    reports: Sequence[Mapping[str, Any]],
    receipts: Sequence[GPUFoundationExecutionReceipt],
) -> bool:
    """Return true only when every accepted shot carries artifact-bound measured QC.

    Production QC evidence must be bound to the exact accepted render receipt, not
    merely contain a syntactically valid SHA-256. This protects the milestone gate
    from stale, swapped, or tampered measurement reports whose artifact digest or
    scene/shot identity belongs to a different render.
    """
    if not reports or len(reports) != len(receipts):
        return False
    for report, receipt in zip(reports, receipts):
        if not isinstance(report, Mapping):
            return False
        if report.get("production_measurement_evidence") is not True:
            return False
        measurement = report.get("measurement")
        if not isinstance(measurement, Mapping):
            return False
        if measurement.get("schema") != "cineos-sequence-quality-measurement/0.1":
            return False
        observer_id = measurement.get("observer_id")
        artifact_sha256 = measurement.get("artifact_sha256")
        if not isinstance(observer_id, str) or not observer_id.strip():
            return False
        if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
            return False
        if artifact_sha256 != receipt.output_sha256:
            return False
        if report.get("output_sha256") != receipt.output_sha256:
            return False
        if report.get("scene_id") != receipt.result.scene_id:
            return False
        if report.get("shot_id") != receipt.result.shot_id:
            return False
        if report.get("effective_request_hash") != receipt.result.request_hash:
            return False
    return True


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

    @property
    def production_gpu_evidence(self) -> bool:
        return _production_gpu_evidence(self.shot_receipts)

    @property
    def production_quality_evidence(self) -> bool:
        return _production_quality_evidence(self.quality_reports, self.shot_receipts)

    @property
    def evidence_tier(self) -> str:
        if self.production_gpu_evidence and self.production_quality_evidence:
            return "production-gpu-quality-gated"
        if self.production_gpu_evidence:
            return "production-gpu-execution"
        return "non-production-or-injected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cineos-gpu-connected-benchmark/0.2",
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
            "production_gpu_evidence": self.production_gpu_evidence,
            "production_quality_evidence": self.production_quality_evidence,
            "evidence_tier": self.evidence_tier,
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
    """Require an explicit linear handoff instead of accepting unrelated clips."""
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
    """Reject recycled output masquerading as a connected multi-shot render."""
    try:
        artifact_path = str(Path(receipt.result.output_path).resolve(strict=False))
    except OSError as exc:
        raise GPUConnectedBenchmarkError(
            f"cannot resolve connected benchmark artifact: {receipt.result.output_path}"
        ) from exc

    if artifact_path in seen_paths:
        raise GPUConnectedBenchmarkError(
            f"connected GPU benchmark reused output path: {artifact_path}"
        )
    if receipt.output_sha256 in seen_hashes:
        raise GPUConnectedBenchmarkError(
            "connected GPU benchmark reused identical rendered bytes across shots"
        )
    seen_paths.add(artifact_path)
    seen_hashes.add(receipt.output_sha256)


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
) -> GPUConnectedBenchmarkReceipt:
    """Render 5-10 connected shots and persist a manifest only on total success."""

    if not benchmark_id.strip():
        raise GPUConnectedBenchmarkError("benchmark_id must not be empty")
    _validate_requests(requests)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / f"{benchmark_id}.gpu-benchmark.json"
    _remove_stale_manifest(manifest)

    receipts: list[GPUFoundationExecutionReceipt] = []
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
                    "shot receipt request hash does not match the current benchmark request"
                )
            _validate_unique_render_evidence(
                receipt,
                seen_paths=seen_paths,
                seen_hashes=seen_hashes,
            )
            receipts.append(receipt)
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
            f"cannot persist connected GPU benchmark evidence: {manifest}"
        ) from exc

    return completed


__all__ = [
    "GPUConnectedBenchmarkError",
    "GPUConnectedBenchmarkReceipt",
    "run_connected_gpu_benchmark",
]
