"""Assemble a production film directly from accepted connected benchmark evidence.

This module narrows the trust boundary between Atlas quality-first GPU execution and
film assembly. Callers no longer need to manufacture per-shot production evidence
records by hand: the records are deterministically derived from the accepted benchmark
receipt and its artifact-bound quality reports.

External pretrained video/QC foundations remain external foundations. This adapter
only binds CINEOS-owned execution/QC evidence to the final assembly path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cineos.atlas.gpu_connected_benchmark import GPUConnectedBenchmarkReceipt

from .exceptions import AssemblyError
from .production_assembly import assemble_production_film


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _required_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise AssemblyError(f"connected benchmark is missing {field}")
    normalized = value.strip().lower()
    if len(normalized) != 64:
        raise AssemblyError(f"connected benchmark has invalid {field}")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise AssemblyError(f"connected benchmark has invalid {field}") from exc
    return normalized


def _validate_benchmark_aggregate_integrity(
    benchmark: GPUConnectedBenchmarkReceipt,
    receipts: Sequence[Any],
) -> None:
    """Bind aggregate benchmark provenance to the exact per-shot receipts.

    Production film assembly must not accept a receipt object whose aggregate chain,
    byte count, profile, or external-origin declaration has drifted from the shots it
    contains. The connected benchmark writes these values at execution time; this
    independent recomputation closes the later assembly trust boundary.
    """

    profile_id = getattr(benchmark, "profile_id", None)
    origin = getattr(benchmark, "origin", None)
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise AssemblyError("connected benchmark is missing profile_id")
    if not isinstance(origin, str) or not origin.strip():
        raise AssemblyError("connected benchmark is missing origin")
    profile_id = profile_id.strip()
    origin = origin.strip()

    digest = hashlib.sha256()
    total_output_bytes = 0
    for index, receipt in enumerate(receipts):
        if getattr(receipt, "profile_id", None) != profile_id:
            raise AssemblyError(
                f"connected benchmark shot {index} profile does not match aggregate profile"
            )
        if getattr(receipt, "origin", None) != origin:
            raise AssemblyError(
                f"connected benchmark shot {index} origin does not match aggregate origin"
            )

        result = getattr(receipt, "result", None)
        request_hash = _required_sha256(
            getattr(result, "request_hash", None),
            field=f"shot {index} request SHA-256",
        )
        output_hash = _required_sha256(
            getattr(receipt, "output_sha256", None),
            field=f"shot {index} output SHA-256",
        )
        digest.update(request_hash.encode("ascii"))
        digest.update(b"\0")
        digest.update(output_hash.encode("ascii"))
        digest.update(b"\n")

        output_bytes = getattr(receipt, "output_bytes", None)
        if (
            isinstance(output_bytes, bool)
            or not isinstance(output_bytes, int)
            or output_bytes <= 0
        ):
            raise AssemblyError(
                f"connected benchmark shot {index} has invalid output byte count"
            )
        total_output_bytes += output_bytes

    expected_chain = digest.hexdigest()
    declared_chain = _required_sha256(
        getattr(benchmark, "chain_sha256", None),
        field="aggregate chain SHA-256",
    )
    if declared_chain != expected_chain:
        raise AssemblyError(
            "connected benchmark aggregate chain does not match contained shot receipts"
        )

    declared_total = getattr(benchmark, "total_output_bytes", None)
    if (
        isinstance(declared_total, bool)
        or not isinstance(declared_total, int)
        or declared_total != total_output_bytes
    ):
        raise AssemblyError(
            "connected benchmark aggregate output byte count does not match contained shots"
        )


def build_production_shot_evidence(
    benchmark: GPUConnectedBenchmarkReceipt,
) -> tuple[dict[str, Any], ...]:
    """Derive assembly records from the exact accepted production benchmark.

    The quality-report canonical digest becomes the assembly evidence digest. Each
    report must independently bind to the same shot id and rendered artifact hash as
    its execution receipt. The adapter fails closed before FFmpeg if any linkage is
    absent, stale, or substituted.
    """

    if benchmark.evidence_tier != "production-gpu-quality-gated":
        raise AssemblyError(
            "production film assembly requires production-gpu-quality-gated benchmark evidence"
        )
    if (
        not benchmark.production_gpu_evidence
        or not benchmark.production_quality_evidence
    ):
        raise AssemblyError(
            "production film assembly requires real GPU execution and measured QC evidence"
        )

    receipts = tuple(benchmark.shot_receipts)
    reports = tuple(benchmark.quality_reports)
    if not 5 <= len(receipts) <= 10:
        raise AssemblyError("connected benchmark must contain 5 to 10 accepted shots")
    if len(reports) != len(receipts):
        raise AssemblyError(
            "connected benchmark quality report count does not match shots"
        )
    _validate_benchmark_aggregate_integrity(benchmark, receipts)

    records: list[dict[str, Any]] = []
    seen_shot_ids: set[str] = set()
    seen_outputs: set[str] = set()
    seen_evidence: set[str] = set()

    for index, (receipt, report) in enumerate(zip(receipts, reports)):
        result = getattr(receipt, "result", None)
        shot_id = getattr(result, "shot_id", None)
        output_path = getattr(result, "output_path", None)
        if not isinstance(shot_id, str) or not shot_id.strip():
            raise AssemblyError(f"connected benchmark shot {index} is missing shot_id")
        shot_id = shot_id.strip()
        if shot_id in seen_shot_ids:
            raise AssemblyError(f"connected benchmark reuses shot_id {shot_id!r}")
        seen_shot_ids.add(shot_id)
        if not isinstance(output_path, str) or not output_path.strip():
            raise AssemblyError(
                f"connected benchmark shot {shot_id} is missing output_path"
            )

        output_sha = _required_sha256(
            getattr(receipt, "output_sha256", None),
            field=f"shot {shot_id} output SHA-256",
        )
        if output_sha in seen_outputs:
            raise AssemblyError(
                "connected benchmark reuses a rendered payload across shots"
            )
        seen_outputs.add(output_sha)

        if not isinstance(report, Mapping) or report.get("accepted") is not True:
            raise AssemblyError(
                f"connected benchmark shot {shot_id} lacks accepted QC evidence"
            )
        if report.get("production_measurement_evidence") is not True:
            raise AssemblyError(
                f"connected benchmark shot {shot_id} lacks measured production QC evidence"
            )
        if report.get("shot_id") != shot_id:
            raise AssemblyError(
                f"connected benchmark shot {shot_id} QC report is bound to another shot"
            )
        if report.get("output_sha256") != output_sha:
            raise AssemblyError(
                f"connected benchmark shot {shot_id} QC report is bound to another render"
            )
        measurement = report.get("measurement")
        if not isinstance(measurement, Mapping):
            raise AssemblyError(
                f"connected benchmark shot {shot_id} lacks artifact-bound QC measurement"
            )
        if measurement.get("artifact_sha256") != output_sha:
            raise AssemblyError(
                f"connected benchmark shot {shot_id} QC measurement is bound to another render"
            )

        evidence_sha = _canonical_sha256(report)
        if evidence_sha in seen_evidence:
            raise AssemblyError("connected benchmark reuses one QC report across shots")
        seen_evidence.add(evidence_sha)
        records.append(
            {
                "shot_id": shot_id,
                "accepted": True,
                "decision": "accept",
                "production_gpu_evidence": True,
                "output_path": output_path,
                "output_sha256": output_sha,
                "evidence_sha256": evidence_sha,
            }
        )

    return tuple(records)


def assemble_benchmark_production_film(
    benchmark: GPUConnectedBenchmarkReceipt,
    output: str | Path,
    *,
    durations: Sequence[float] | None = None,
    audio_path: str | Path | None = None,
    audio_sha256: str | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble an exact quality-first benchmark into a validated production film."""

    return assemble_production_film(
        build_production_shot_evidence(benchmark),
        output,
        durations=durations,
        audio_path=audio_path,
        audio_sha256=audio_sha256,
        manifest_path=manifest_path,
    )


__all__ = ["assemble_benchmark_production_film", "build_production_shot_evidence"]
