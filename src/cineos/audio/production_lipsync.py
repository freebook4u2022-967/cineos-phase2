"""Fail-closed production evidence for dialogue lip-sync.

This module validates measured lip-sync evidence without claiming that timing or
alignment was produced by a CINEOS-native model. External/open scorers may be
used when their provenance is declared; production evidence is accepted only
when it is cryptographically bound to the exact rendered shot and approved
spoken-audio artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class ProductionLipSyncError(RuntimeError):
    """Raised when dialogue lip-sync evidence is incomplete or untrustworthy."""


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ProductionLipSyncError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProductionLipSyncError(f"{name} must be a SHA-256 hex digest") from exc
    return value.lower()


@dataclass(frozen=True, slots=True)
class ProductionLipSyncPolicy:
    """Thresholds for accepting measured dialogue synchronization evidence."""

    minimum_sync_confidence: float = 0.78
    maximum_mean_offset_ms: float = 120.0
    maximum_p95_offset_ms: float = 220.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_sync_confidence <= 1.0:
            raise ValueError("minimum_sync_confidence must be between 0 and 1")
        if self.maximum_mean_offset_ms < 0 or self.maximum_p95_offset_ms < 0:
            raise ValueError("offset thresholds must be non-negative")
        if self.maximum_p95_offset_ms < self.maximum_mean_offset_ms:
            raise ValueError("maximum_p95_offset_ms must be >= maximum_mean_offset_ms")


@dataclass(frozen=True, slots=True)
class ProductionLipSyncEvidence:
    schema: str
    shot_id: str
    character_id: str
    dialogue_cue_id: str
    rendered_video_sha256: str
    dialogue_audio_sha256: str
    scorer_name: str
    scorer_provenance: str
    sync_confidence: float
    mean_offset_ms: float
    p95_offset_ms: float
    measured_frame_count: int
    measured_word_count: int
    accepted: bool
    decision: str
    failed_metrics: tuple[str, ...]
    evidence_sha256: str


def validate_production_lipsync(
    report: Mapping[str, Any],
    *,
    rendered_video_path: str | Path,
    dialogue_audio_path: str | Path,
    shot_id: str,
    character_id: str,
    dialogue_cue_id: str,
    policy: ProductionLipSyncPolicy | None = None,
) -> ProductionLipSyncEvidence:
    """Validate scorer output and bind it to exact production artifacts.

    ``report`` must come from a scorer that actually inspected the rendered
    video and approved dialogue audio. Synthetic/default metrics are rejected by
    requiring declared scorer provenance and non-zero measured sample counts.
    """

    policy = policy or ProductionLipSyncPolicy()
    video = Path(rendered_video_path)
    audio = Path(dialogue_audio_path)
    if not video.is_file():
        raise ProductionLipSyncError("rendered video artifact does not exist")
    if not audio.is_file():
        raise ProductionLipSyncError("dialogue audio artifact does not exist")

    expected_video_hash = _sha256(video)
    expected_audio_hash = _sha256(audio)
    if _require_hash(report.get("rendered_video_sha256"), "rendered_video_sha256") != expected_video_hash:
        raise ProductionLipSyncError("lip-sync report is not bound to the exact rendered video artifact")
    if _require_hash(report.get("dialogue_audio_sha256"), "dialogue_audio_sha256") != expected_audio_hash:
        raise ProductionLipSyncError("lip-sync report is not bound to the exact approved dialogue audio artifact")

    identifiers = {
        "shot_id": shot_id,
        "character_id": character_id,
        "dialogue_cue_id": dialogue_cue_id,
    }
    for name, expected in identifiers.items():
        value = report.get(name)
        if not isinstance(value, str) or value != expected:
            raise ProductionLipSyncError(f"lip-sync report {name} does not match the production request")

    scorer_name = report.get("scorer_name")
    scorer_provenance = report.get("scorer_provenance")
    if not isinstance(scorer_name, str) or not scorer_name.strip():
        raise ProductionLipSyncError("production lip-sync evidence requires a declared scorer_name")
    if not isinstance(scorer_provenance, str) or not scorer_provenance.strip():
        raise ProductionLipSyncError("production lip-sync evidence requires declared scorer_provenance")

    try:
        confidence = float(report["sync_confidence"])
        mean_offset = float(report["mean_offset_ms"])
        p95_offset = float(report["p95_offset_ms"])
        measured_frames = int(report["measured_frame_count"])
        measured_words = int(report["measured_word_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProductionLipSyncError("lip-sync report is missing valid measured metrics") from exc

    if not 0.0 <= confidence <= 1.0:
        raise ProductionLipSyncError("sync_confidence must be between 0 and 1")
    if mean_offset < 0 or p95_offset < 0 or p95_offset < mean_offset:
        raise ProductionLipSyncError("lip-sync offset metrics are invalid")
    if measured_frames <= 0 or measured_words <= 0:
        raise ProductionLipSyncError("production lip-sync evidence requires measured frames and words")

    failures: list[str] = []
    if confidence < policy.minimum_sync_confidence:
        failures.append("sync_confidence")
    if mean_offset > policy.maximum_mean_offset_ms:
        failures.append("mean_offset_ms")
    if p95_offset > policy.maximum_p95_offset_ms:
        failures.append("p95_offset_ms")

    accepted = not failures
    payload = {
        "schema": "cineos-production-lipsync-evidence/0.1",
        "shot_id": shot_id,
        "character_id": character_id,
        "dialogue_cue_id": dialogue_cue_id,
        "rendered_video_sha256": expected_video_hash,
        "dialogue_audio_sha256": expected_audio_hash,
        "scorer_name": scorer_name.strip(),
        "scorer_provenance": scorer_provenance.strip(),
        "sync_confidence": confidence,
        "mean_offset_ms": mean_offset,
        "p95_offset_ms": p95_offset,
        "measured_frame_count": measured_frames,
        "measured_word_count": measured_words,
        "accepted": accepted,
        "decision": "accept" if accepted else "reject",
        "failed_metrics": tuple(failures),
    }
    evidence_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ProductionLipSyncEvidence(**payload, evidence_sha256=evidence_hash)


def evidence_manifest(evidence: ProductionLipSyncEvidence) -> dict[str, Any]:
    """Return a JSON-serializable evidence object with its integrity hash."""

    return asdict(evidence)


__all__ = [
    "ProductionLipSyncError",
    "ProductionLipSyncEvidence",
    "ProductionLipSyncPolicy",
    "evidence_manifest",
    "validate_production_lipsync",
]
