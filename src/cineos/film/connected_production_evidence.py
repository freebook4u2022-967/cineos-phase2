"""Cross-bind connected GPU benchmark evidence to the assembled production film.

This module closes the evidence boundary between Atlas connected-shot validation and
film assembly. It does not make an external pretrained video foundation CINEOS-native;
it proves that the exact GPU/QC-approved render artifacts accepted by the connected
benchmark are the artifacts represented in the production assembly manifest, that
required dialogue shots carry measured lip-sync QC bound to the exact video and
approved dialogue audio, that those same dialogue artifacts are inputs to the exact
approved production audio mix, and that the final MP4 still matches its recorded digest.
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
from cineos.audio.lipsync_qc import (
    LipSyncAnalyzerProvenance,
    LipSyncQCError,
    LipSyncThresholds,
    validate_lipsync_quality_evidence,
)
from cineos.audio.production_mix_evidence import (
    ProductionAudioMixEvidenceError,
    validate_production_audio_mix_evidence,
)

from .production_assembly import PRODUCTION_EVIDENCE_SCHEMA
from .validator import file_hash

CONNECTED_PRODUCTION_FILM_EVIDENCE_SCHEMA = (
    "cineos-connected-production-film-evidence/0.6"
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
    dialogue_shot_ids: tuple[str, ...] = ()
    lipsync_evidence_sha256: tuple[str, ...] = ()
    audio_mix_evidence_sha256: str | None = None

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
            "dialogue_shot_ids": list(self.dialogue_shot_ids),
            "lipsync_evidence_sha256": list(self.lipsync_evidence_sha256),
            "audio_mix_evidence_sha256": self.audio_mix_evidence_sha256,
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


def _validate_benchmark_identity(benchmark: GPUConnectedBenchmarkReceipt) -> None:
    _required_text(benchmark.benchmark_id, field="benchmark ID")
    _required_text(benchmark.profile_id, field="foundation profile ID")
    origin = _required_text(benchmark.origin, field="foundation origin")
    if origin != "external_pretrained_foundation":
        raise ConnectedProductionFilmEvidenceError(
            "connected production film evidence must identify the video model as an "
            "external_pretrained_foundation"
        )


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

    seen_evidence_hashes: set[str] = set()
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

        evidence_sha = _required_sha256(
            shot.get("evidence_sha256"),
            field=f"assembly shot {index} evidence SHA-256",
        )
        if evidence_sha in seen_evidence_hashes:
            raise ConnectedProductionFilmEvidenceError(
                f"production assembly shot {index} reuses QC evidence from another shot"
            )
        seen_evidence_hashes.add(evidence_sha)

        if shot.get("index") != index:
            raise ConnectedProductionFilmEvidenceError(
                f"production assembly shot {index} has invalid timeline index"
            )


def _normalize_dialogue_shot_ids(values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        shot_id = _required_text(value, field=f"dialogue shot ID {index}")
        if shot_id in seen:
            raise ConnectedProductionFilmEvidenceError(
                f"duplicate required dialogue shot ID: {shot_id}"
            )
        seen.add(shot_id)
        normalized.append(shot_id)
    return tuple(normalized)


def _resolve_dialogue_shot_ids(
    benchmark: GPUConnectedBenchmarkReceipt,
    requested: Sequence[str],
) -> tuple[str, ...]:
    """Resolve authoritative dialogue scope from the exact rendered benchmark."""
    explicit = _normalize_dialogue_shot_ids(requested)
    scope_declared = getattr(benchmark, "dialogue_scope_declared", None)
    declared_values = getattr(benchmark, "dialogue_shot_ids", None)

    if scope_declared is None:
        return explicit
    if scope_declared is not True:
        if declared_values not in (None, [], ()) or explicit:
            raise ConnectedProductionFilmEvidenceError(
                "connected benchmark has inconsistent dialogue scope declaration"
            )
        return ()
    if declared_values is None:
        raise ConnectedProductionFilmEvidenceError(
            "connected benchmark declared dialogue scope without dialogue shot IDs"
        )

    declared = _normalize_dialogue_shot_ids(declared_values)
    benchmark_shot_ids = tuple(
        _required_text(
            getattr(getattr(receipt, "result", None), "shot_id", None),
            field=f"benchmark shot {index} ID",
        )
        for index, receipt in enumerate(benchmark.shot_receipts)
    )
    unknown = set(declared).difference(benchmark_shot_ids)
    if unknown:
        raise ConnectedProductionFilmEvidenceError(
            "connected benchmark dialogue scope references unknown rendered shot: "
            + ", ".join(sorted(unknown))
        )

    if explicit and explicit != declared:
        raise ConnectedProductionFilmEvidenceError(
            "caller dialogue scope conflicts with connected benchmark declaration"
        )
    return declared


def _validate_lipsync_binding(
    assembly: Mapping[str, Any],
    *,
    lipsync_evidence: Sequence[Mapping[str, Any]] | None,
    required_dialogue_shot_ids: Sequence[str],
    dialogue_audio_sha256_by_shot: Mapping[str, str] | None,
    expected_lipsync_analyzer: LipSyncAnalyzerProvenance | None,
    lipsync_thresholds: LipSyncThresholds | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required = _normalize_dialogue_shot_ids(required_dialogue_shot_ids)
    reports = tuple(lipsync_evidence or ())
    if not required:
        if reports:
            raise ConnectedProductionFilmEvidenceError(
                "lip-sync evidence was supplied without required dialogue shot IDs"
            )
        return (), ()

    if not reports:
        raise ConnectedProductionFilmEvidenceError(
            "dialogue-bearing production film requires measured lip-sync evidence"
        )
    if dialogue_audio_sha256_by_shot is None:
        raise ConnectedProductionFilmEvidenceError(
            "dialogue-bearing production film requires approved dialogue audio hashes"
        )
    if expected_lipsync_analyzer is None:
        raise ConnectedProductionFilmEvidenceError(
            "dialogue-bearing production film requires pinned lip-sync analyzer provenance"
        )

    shot_hashes: dict[str, str] = {}
    for index, shot in enumerate(_assembly_shots(assembly)):
        shot_id = _required_text(shot.get("shot_id"), field=f"assembly shot {index} ID")
        shot_hashes[shot_id] = _required_sha256(
            shot.get("output_sha256"),
            field=f"assembly shot {index} output SHA-256",
        )

    required_set = set(required)
    missing_from_assembly = required_set.difference(shot_hashes)
    if missing_from_assembly:
        raise ConnectedProductionFilmEvidenceError(
            "required dialogue shot is absent from production assembly: "
            + ", ".join(sorted(missing_from_assembly))
        )

    validated_hashes: dict[str, str] = {}
    for index, report in enumerate(reports):
        if not isinstance(report, Mapping):
            raise ConnectedProductionFilmEvidenceError(
                f"lip-sync evidence {index} must be a mapping"
            )
        shot_id = _required_text(
            report.get("shot_id"), field=f"lip-sync evidence {index} shot ID"
        )
        if shot_id not in required_set:
            raise ConnectedProductionFilmEvidenceError(
                f"lip-sync evidence references non-required dialogue shot: {shot_id}"
            )
        if shot_id in validated_hashes:
            raise ConnectedProductionFilmEvidenceError(
                f"duplicate lip-sync evidence for dialogue shot: {shot_id}"
            )
        expected_audio = _required_sha256(
            dialogue_audio_sha256_by_shot.get(shot_id),
            field=f"dialogue shot {shot_id} approved audio SHA-256",
        )
        try:
            validate_lipsync_quality_evidence(
                report,
                expected_shot_id=shot_id,
                expected_video_sha256=shot_hashes[shot_id],
                expected_audio_sha256=expected_audio,
                thresholds=lipsync_thresholds,
                expected_analyzer=expected_lipsync_analyzer,
            )
        except (LipSyncQCError, TypeError) as exc:
            raise ConnectedProductionFilmEvidenceError(
                f"dialogue shot {shot_id} has invalid measured lip-sync evidence: {exc}"
            ) from exc
        validated_hashes[shot_id] = _required_sha256(
            report.get("evidence_sha256"),
            field=f"dialogue shot {shot_id} lip-sync evidence SHA-256",
        )

    missing = required_set.difference(validated_hashes)
    if missing:
        raise ConnectedProductionFilmEvidenceError(
            "dialogue-bearing production film is missing measured lip-sync evidence for: "
            + ", ".join(sorted(missing))
        )
    return required, tuple(validated_hashes[shot_id] for shot_id in required)


def _validate_dialogue_mix_binding(
    benchmark: GPUConnectedBenchmarkReceipt,
    assembly: Mapping[str, Any],
    *,
    dialogue_shot_ids: Sequence[str],
    dialogue_audio_sha256_by_shot: Mapping[str, str] | None,
    audio_mix_evidence: Mapping[str, Any] | None,
) -> str | None:
    """Bind modern authoritative dialogue scope to the exact final assembly mix."""
    if getattr(benchmark, "dialogue_scope_declared", None) is not True:
        return None
    required = _normalize_dialogue_shot_ids(dialogue_shot_ids)
    if not required:
        return None
    if dialogue_audio_sha256_by_shot is None:
        raise ConnectedProductionFilmEvidenceError(
            "dialogue-bearing production film requires approved dialogue audio hashes"
        )
    if not isinstance(audio_mix_evidence, Mapping):
        raise ConnectedProductionFilmEvidenceError(
            "GPU-declared dialogue film requires production audio mix evidence"
        )
    try:
        mix_output_sha = validate_production_audio_mix_evidence(audio_mix_evidence)
    except (ProductionAudioMixEvidenceError, TypeError) as exc:
        raise ConnectedProductionFilmEvidenceError(
            f"production audio mix evidence is invalid: {exc}"
        ) from exc

    assembly_audio = assembly.get("audio")
    if not isinstance(assembly_audio, Mapping):
        raise ConnectedProductionFilmEvidenceError(
            "GPU-declared dialogue film requires an approved assembly audio artifact"
        )
    assembly_audio_sha = _required_sha256(
        assembly_audio.get("sha256"), field="assembly audio SHA-256"
    )
    if mix_output_sha != assembly_audio_sha:
        raise ConnectedProductionFilmEvidenceError(
            "production audio mix output does not match approved assembly audio"
        )

    inputs = audio_mix_evidence.get("inputs")
    if not isinstance(inputs, list):
        raise ConnectedProductionFilmEvidenceError(
            "production audio mix evidence requires an input list"
        )
    dialogue_inputs: dict[str, str] = {}
    for index, item in enumerate(inputs):
        if not isinstance(item, Mapping) or item.get("kind") != "dialogue":
            continue
        shot_id = _required_text(
            item.get("shot_id"), field=f"dialogue mix input {index} shot ID"
        )
        if shot_id in dialogue_inputs:
            raise ConnectedProductionFilmEvidenceError(
                f"duplicate dialogue mix input for shot: {shot_id}"
            )
        dialogue_inputs[shot_id] = _required_sha256(
            item.get("sha256"), field=f"dialogue mix input {index} SHA-256"
        )

    required_set = set(required)
    actual_set = set(dialogue_inputs)
    if actual_set != required_set:
        missing = required_set.difference(actual_set)
        extra = actual_set.difference(required_set)
        detail: list[str] = []
        if missing:
            detail.append("missing " + ", ".join(sorted(missing)))
        if extra:
            detail.append("unexpected " + ", ".join(sorted(extra)))
        raise ConnectedProductionFilmEvidenceError(
            "production audio mix dialogue scope does not match connected benchmark: "
            + "; ".join(detail)
        )
    for shot_id in required:
        expected = _required_sha256(
            dialogue_audio_sha256_by_shot.get(shot_id),
            field=f"dialogue shot {shot_id} approved audio SHA-256",
        )
        if dialogue_inputs[shot_id] != expected:
            raise ConnectedProductionFilmEvidenceError(
                f"production audio mix substitutes dialogue audio for shot: {shot_id}"
            )
    return _required_sha256(
        audio_mix_evidence.get("evidence_sha256"), field="audio mix evidence SHA-256"
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
    *,
    lipsync_evidence: Sequence[Mapping[str, Any]] | None = None,
    required_dialogue_shot_ids: Sequence[str] = (),
    dialogue_audio_sha256_by_shot: Mapping[str, str] | None = None,
    expected_lipsync_analyzer: LipSyncAnalyzerProvenance | None = None,
    lipsync_thresholds: LipSyncThresholds | None = None,
    audio_mix_evidence: Mapping[str, Any] | None = None,
) -> ConnectedProductionFilmEvidence:
    """Require one exact evidence chain from connected GPU renders to final MP4.

    Legacy receipts retain their previous contract. GPU benchmark receipts that
    authoritatively declare dialogue must additionally prove that the same exact
    dialogue artifacts used for measured lip-sync QC were consumed by the exact audio
    mix supplied to production assembly.
    """
    if not isinstance(benchmark, GPUConnectedBenchmarkReceipt):
        raise TypeError("benchmark must be a GPUConnectedBenchmarkReceipt")
    if not isinstance(assembly, Mapping):
        raise TypeError("assembly must be a mapping")
    if assembly.get("schema") != PRODUCTION_EVIDENCE_SCHEMA:
        raise ConnectedProductionFilmEvidenceError(
            "production assembly has unsupported evidence schema"
        )

    _validate_benchmark_identity(benchmark)
    try:
        connected = validate_production_connected_evidence(benchmark)
    except ProductionConnectedEvidenceError as exc:
        raise ConnectedProductionFilmEvidenceError(
            f"connected production benchmark evidence is invalid: {exc}"
        ) from exc

    manifest_sha = _validate_manifest_integrity(assembly)
    _validate_shot_binding(benchmark, assembly)
    authoritative_dialogue_shot_ids = _resolve_dialogue_shot_ids(
        benchmark, required_dialogue_shot_ids
    )
    dialogue_shot_ids, lipsync_hashes = _validate_lipsync_binding(
        assembly,
        lipsync_evidence=lipsync_evidence,
        required_dialogue_shot_ids=authoritative_dialogue_shot_ids,
        dialogue_audio_sha256_by_shot=dialogue_audio_sha256_by_shot,
        expected_lipsync_analyzer=expected_lipsync_analyzer,
        lipsync_thresholds=lipsync_thresholds,
    )
    mix_evidence_sha = _validate_dialogue_mix_binding(
        benchmark,
        assembly,
        dialogue_shot_ids=dialogue_shot_ids,
        dialogue_audio_sha256_by_shot=dialogue_audio_sha256_by_shot,
        audio_mix_evidence=audio_mix_evidence,
    )
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
        dialogue_shot_ids=dialogue_shot_ids,
        lipsync_evidence_sha256=lipsync_hashes,
        audio_mix_evidence_sha256=mix_evidence_sha,
    )
    if not evidence.accepted:
        raise ConnectedProductionFilmEvidenceError(
            "connected production film evidence did not satisfy the unified gate"
        )
    return evidence


def connected_production_film_evidence(
    benchmark: GPUConnectedBenchmarkReceipt,
    assembly: Mapping[str, Any],
    *,
    lipsync_evidence: Sequence[Mapping[str, Any]] | None = None,
    required_dialogue_shot_ids: Sequence[str] = (),
    dialogue_audio_sha256_by_shot: Mapping[str, str] | None = None,
    expected_lipsync_analyzer: LipSyncAnalyzerProvenance | None = None,
    lipsync_thresholds: LipSyncThresholds | None = None,
    audio_mix_evidence: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether connected benchmark evidence binds to the exact final film."""
    try:
        validate_connected_production_film_evidence(
            benchmark,
            assembly,
            lipsync_evidence=lipsync_evidence,
            required_dialogue_shot_ids=required_dialogue_shot_ids,
            dialogue_audio_sha256_by_shot=dialogue_audio_sha256_by_shot,
            expected_lipsync_analyzer=expected_lipsync_analyzer,
            lipsync_thresholds=lipsync_thresholds,
            audio_mix_evidence=audio_mix_evidence,
        )
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
