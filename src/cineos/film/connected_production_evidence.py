"""Cross-bind connected GPU benchmark evidence to the assembled production film.

This module closes the evidence boundary between Atlas connected-shot validation and
film assembly. It does not make an external pretrained video foundation CINEOS-native;
it proves that the exact GPU/QC-approved render artifacts accepted by the connected
benchmark are the artifacts represented in the production assembly manifest, and that
the final MP4 still matches its recorded digest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cineos.atlas.gpu_connected_benchmark import GPUConnectedBenchmarkReceipt
from cineos.atlas.production_connected_evidence import (
    ProductionConnectedEvidence,
    ProductionConnectedEvidenceError,
    validate_production_connected_evidence,
)

from .production_assembly import PRODUCTION_EVIDENCE_SCHEMA
from .validator import file_hash

CONNECTED_PRODUCTION_FILM_EVIDENCE_SCHEMA = (
    "cineos-connected-production-film-evidence/0.1"
)


class ConnectedProductionFilmEvidenceError(RuntimeError):
    """Raised when connected render evidence does not bind to the final film."""


@dataclass(frozen=True, slots=True)
class ConnectedProductionFilmEvidence:
    """Auditable cross-binding between connected renders and final film assembly."""

    benchmark_id: str
    profile_id: str
    origin: str
    shot_count: int
    benchmark_chain_sha256: str
    assembly_manifest_sha256: str
    final_mp4_sha256: str
    connected_evidence: ProductionConnectedEvidence

    @property
    def accepted(self) -> bool:
        return self.connected_evidence.accepted

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONNECTED_PRODUCTION_FILM_EVIDENCE_SCHEMA,
            "benchmark_id": self.benchmark_id,
            "profile_id": self.profile_id,
            "origin": self.origin,
            "shot_count": self.shot_count,
            "benchmark_chain_sha256": self.benchmark_chain_sha256,
            "assembly_manifest_sha256": self.assembly_manifest_sha256,
            "final_mp4_sha256": self.final_mp4_sha256,
            "accepted": self.accepted,
            "connected_evidence": self.connected_evidence.to_dict(),
        }


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConnectedProductionFilmEvidenceError(
            f"connected production film evidence requires {field}"
        )
    return value.strip()


def _required_sha256(value: Any, *, field: str) -> str:
    normalized = _required_text(value, field=field).lower()
    if len(normalized) != 64:
        raise ConnectedProductionFilmEvidenceError(
            f"connected production film evidence requires a valid {field}"
        )
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ConnectedProductionFilmEvidenceError(
            f"connected production film evidence requires a valid {field}"
        ) from exc
    return normalized


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_manifest_integrity(assembly: Mapping[str, Any]) -> str:
    recorded = _required_sha256(
        assembly.get("manifest_sha256"), field="assembly manifest SHA-256"
    )
    unsigned = dict(assembly)
    unsigned.pop("manifest_sha256", None)
    computed = _canonical_hash(unsigned)
    if computed != recorded:
        raise ConnectedProductionFilmEvidenceError(
            "production assembly manifest SHA-256 does not match its contents"
        )
    return recorded


def _assembly_shots(assembly: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    shots = assembly.get("shots")
    if not isinstance(shots, list) or not all(
        isinstance(item, Mapping) for item in shots
    ):
        raise ConnectedProductionFilmEvidenceError(
            "production assembly requires an ordered shot evidence list"
        )
    return shots


def _validate_shot_binding(
    benchmark: GPUConnectedBenchmarkReceipt,
    assembly: Mapping[str, Any],
) -> None:
    shots = _assembly_shots(assembly)
    if assembly.get("shot_count") != len(benchmark.shot_receipts):
        raise ConnectedProductionFilmEvidenceError(
            "production assembly shot count does not match connected benchmark"
        )
    if len(shots) != len(benchmark.shot_receipts):
        raise ConnectedProductionFilmEvidenceError(
            "production assembly shot list does not match connected benchmark"
        )

    for index, (shot, receipt) in enumerate(zip(shots, benchmark.shot_receipts)):
        result = getattr(receipt, "result", None)
        if result is None:
            raise ConnectedProductionFilmEvidenceError(
                f"connected benchmark shot {index} has no render result"
            )
        benchmark_shot_id = _required_text(
            getattr(result, "shot_id", None), field=f"benchmark shot {index} ID"
        )
        assembly_shot_id = _required_text(
            shot.get("shot_id"), field=f"assembly shot {index} ID"
        )
        if assembly_shot_id != benchmark_shot_id:
            raise ConnectedProductionFilmEvidenceError(
                f"production assembly shot {index} ID does not match connected benchmark"
            )

        benchmark_sha = _required_sha256(
            getattr(receipt, "output_sha256", None),
            field=f"benchmark shot {index} output SHA-256",
        )
        assembly_sha = _required_sha256(
            shot.get("output_sha256"),
            field=f"assembly shot {index} output SHA-256",
        )
        if assembly_sha != benchmark_sha:
            raise ConnectedProductionFilmEvidenceError(
                f"production assembly shot {index} artifact does not match connected benchmark"
            )
        if shot.get("index") != index:
            raise ConnectedProductionFilmEvidenceError(
                f"production assembly shot {index} has invalid timeline index"
            )


def _validate_final_artifact(assembly: Mapping[str, Any]) -> str:
    expected = _required_sha256(
        assembly.get("final_mp4_sha256"), field="final MP4 SHA-256"
    )
    movie = Path(
        _required_text(assembly.get("final_mp4"), field="final MP4 path")
    ).resolve()
    try:
        actual = file_hash(movie)
    except (OSError, ValueError) as exc:
        raise ConnectedProductionFilmEvidenceError(
            "connected production final MP4 cannot be verified"
        ) from exc
    if actual != expected:
        raise ConnectedProductionFilmEvidenceError(
            "connected production final MP4 hash does not match assembly evidence"
        )
    return expected


def validate_connected_production_film_evidence(
    benchmark: GPUConnectedBenchmarkReceipt,
    assembly: Mapping[str, Any],
) -> ConnectedProductionFilmEvidence:
    """Require one exact evidence chain from connected GPU renders to final MP4."""

    if not isinstance(benchmark, GPUConnectedBenchmarkReceipt):
        raise TypeError("benchmark must be a GPUConnectedBenchmarkReceipt")
    if not isinstance(assembly, Mapping):
        raise TypeError("assembly must be a mapping")
    if assembly.get("schema") != PRODUCTION_EVIDENCE_SCHEMA:
        raise ConnectedProductionFilmEvidenceError(
            "production assembly has unsupported evidence schema"
        )

    try:
        connected = validate_production_connected_evidence(benchmark)
    except ProductionConnectedEvidenceError as exc:
        raise ConnectedProductionFilmEvidenceError(
            f"connected production benchmark evidence is invalid: {exc}"
        ) from exc

    manifest_sha = _validate_manifest_integrity(assembly)
    _validate_shot_binding(benchmark, assembly)
    final_sha = _validate_final_artifact(assembly)

    evidence = ConnectedProductionFilmEvidence(
        benchmark_id=benchmark.benchmark_id,
        profile_id=benchmark.profile_id,
        origin=benchmark.origin,
        shot_count=len(benchmark.shot_receipts),
        benchmark_chain_sha256=benchmark.chain_sha256,
        assembly_manifest_sha256=manifest_sha,
        final_mp4_sha256=final_sha,
        connected_evidence=connected,
    )
    if not evidence.accepted:
        raise ConnectedProductionFilmEvidenceError(
            "connected production film evidence did not satisfy the unified gate"
        )
    return evidence


def connected_production_film_evidence(
    benchmark: GPUConnectedBenchmarkReceipt,
    assembly: Mapping[str, Any],
) -> bool:
    """Return whether connected benchmark evidence binds to the exact final film."""

    try:
        validate_connected_production_film_evidence(benchmark, assembly)
    except (ConnectedProductionFilmEvidenceError, TypeError):
        return False
    return True


__all__ = [
    "CONNECTED_PRODUCTION_FILM_EVIDENCE_SCHEMA",
    "ConnectedProductionFilmEvidence",
    "ConnectedProductionFilmEvidenceError",
    "connected_production_film_evidence",
    "validate_connected_production_film_evidence",
]
