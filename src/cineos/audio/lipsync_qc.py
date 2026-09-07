"""Measured audio/visual lip-sync QC for production dialogue shots.

This module deliberately does not claim that CINEOS natively solves lip synchronization.
It provides a fail-closed contract around an external learned analyzer and binds its
measurements to the exact video/audio artifacts that were evaluated.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LIPSYNC_QC_SCHEMA = "cineos-lipsync-qc/0.1"
ALLOWED_ANALYZER_ORIGINS = frozenset(
    {"external_pretrained_foundation", "external_reference_analyzer"}
)


class LipSyncQCError(RuntimeError):
    """Raised when measured lip-sync evidence is missing, malformed, or rejected."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LipSyncQCError(f"cannot hash lip-sync artifact: {path}") from exc
    return digest.hexdigest()


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LipSyncQCError(f"lip-sync QC requires {field}")
    return value.strip()


def _required_sha256(value: Any, *, field: str) -> str:
    normalized = _required_text(value, field=field).lower()
    if len(normalized) != 64:
        raise LipSyncQCError(f"lip-sync QC requires a valid {field}")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise LipSyncQCError(f"lip-sync QC requires a valid {field}") from exc
    return normalized


def _finite(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LipSyncQCError(f"lip-sync QC requires finite {field}") from exc
    if not math.isfinite(number):
        raise LipSyncQCError(f"lip-sync QC requires finite {field}")
    return number


def _unit_interval(value: Any, *, field: str) -> float:
    number = _finite(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise LipSyncQCError(f"lip-sync QC {field} must be within [0, 1]")
    return number


@dataclass(frozen=True, slots=True)
class LipSyncThresholds:
    """Release thresholds for measured dialogue synchronization."""

    min_sync_confidence: float = 0.65
    max_abs_offset_ms: float = 120.0
    min_speaking_frame_coverage: float = 0.60
    min_face_track_coverage: float = 0.90

    def validated(self) -> LipSyncThresholds:
        confidence = _unit_interval(
            self.min_sync_confidence, field="minimum sync confidence"
        )
        offset = _finite(self.max_abs_offset_ms, field="maximum absolute offset")
        if offset < 0:
            raise LipSyncQCError(
                "lip-sync QC maximum absolute offset must be non-negative"
            )
        speaking = _unit_interval(
            self.min_speaking_frame_coverage,
            field="minimum speaking-frame coverage",
        )
        face = _unit_interval(
            self.min_face_track_coverage, field="minimum face-track coverage"
        )
        return LipSyncThresholds(confidence, offset, speaking, face)


@dataclass(frozen=True, slots=True)
class LipSyncQualityEvidence:
    """Artifact-bound measurements from one learned A/V synchronization analyzer."""

    shot_id: str
    video_sha256: str
    audio_sha256: str
    analyzer_origin: str
    analyzer_id: str
    analyzer_revision: str
    analyzer_license_id: str
    analyzer_source_url: str
    sync_confidence: float
    av_offset_ms: float
    speaking_frame_coverage: float
    face_track_coverage: float
    measured: bool = True
    accepted: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = {"schema": LIPSYNC_QC_SCHEMA, **asdict(self)}
        payload["evidence_sha256"] = _canonical_hash(payload)
        return payload


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_lipsync_quality_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_shot_id: str,
    expected_video_sha256: str,
    expected_audio_sha256: str,
    thresholds: LipSyncThresholds | None = None,
) -> LipSyncQualityEvidence:
    """Validate measured evidence and bind it to exact dialogue artifacts."""

    if not isinstance(evidence, Mapping):
        raise TypeError("lip-sync evidence must be a mapping")
    if evidence.get("schema") != LIPSYNC_QC_SCHEMA:
        raise LipSyncQCError("unsupported lip-sync QC evidence schema")
    if evidence.get("measured") is not True:
        raise LipSyncQCError("production dialogue requires measured lip-sync QC")

    shot_id = _required_text(evidence.get("shot_id"), field="shot ID")
    if shot_id != _required_text(expected_shot_id, field="expected shot ID"):
        raise LipSyncQCError("lip-sync QC shot ID does not match dialogue shot")

    video_sha = _required_sha256(evidence.get("video_sha256"), field="video SHA-256")
    audio_sha = _required_sha256(evidence.get("audio_sha256"), field="audio SHA-256")
    if video_sha != _required_sha256(
        expected_video_sha256, field="expected video SHA-256"
    ):
        raise LipSyncQCError("lip-sync QC video artifact does not match dialogue shot")
    if audio_sha != _required_sha256(
        expected_audio_sha256, field="expected audio SHA-256"
    ):
        raise LipSyncQCError("lip-sync QC audio artifact does not match dialogue audio")

    origin = _required_text(evidence.get("analyzer_origin"), field="analyzer origin")
    if origin not in ALLOWED_ANALYZER_ORIGINS:
        raise LipSyncQCError(
            "lip-sync analyzer must be identified as an external learned/reference "
            "foundation; it cannot be relabeled as CINEOS-native"
        )

    analyzer_id = _required_text(evidence.get("analyzer_id"), field="analyzer ID")
    revision = _required_text(
        evidence.get("analyzer_revision"), field="analyzer revision"
    )
    license_id = _required_text(
        evidence.get("analyzer_license_id"), field="analyzer license ID"
    )
    source_url = _required_text(
        evidence.get("analyzer_source_url"), field="analyzer source URL"
    )

    confidence = _unit_interval(
        evidence.get("sync_confidence"), field="sync confidence"
    )
    offset = _finite(evidence.get("av_offset_ms"), field="A/V offset")
    speaking = _unit_interval(
        evidence.get("speaking_frame_coverage"), field="speaking-frame coverage"
    )
    face = _unit_interval(
        evidence.get("face_track_coverage"), field="face-track coverage"
    )
    limits = (thresholds or LipSyncThresholds()).validated()
    accepted = (
        confidence >= limits.min_sync_confidence
        and abs(offset) <= limits.max_abs_offset_ms
        and speaking >= limits.min_speaking_frame_coverage
        and face >= limits.min_face_track_coverage
    )
    if evidence.get("accepted") is not accepted:
        raise LipSyncQCError(
            "lip-sync QC accepted flag does not match measured release thresholds"
        )
    if not accepted:
        raise LipSyncQCError("dialogue shot failed measured lip-sync QC")

    unsigned = dict(evidence)
    recorded_hash = _required_sha256(
        unsigned.pop("evidence_sha256", None), field="evidence SHA-256"
    )
    if _canonical_hash(unsigned) != recorded_hash:
        raise LipSyncQCError("lip-sync QC evidence SHA-256 does not match its contents")

    return LipSyncQualityEvidence(
        shot_id=shot_id,
        video_sha256=video_sha,
        audio_sha256=audio_sha,
        analyzer_origin=origin,
        analyzer_id=analyzer_id,
        analyzer_revision=revision,
        analyzer_license_id=license_id,
        analyzer_source_url=source_url,
        sync_confidence=confidence,
        av_offset_ms=offset,
        speaking_frame_coverage=speaking,
        face_track_coverage=face,
        measured=True,
        accepted=True,
    )


@dataclass(frozen=True, slots=True)
class ExternalLipSyncAnalyzer:
    """Invoke a configured external learned analyzer without disguising provenance."""

    command: tuple[str, ...]
    origin: str = "external_pretrained_foundation"
    timeout_seconds: float = 300.0

    def measure(
        self,
        *,
        shot_id: str,
        video_path: str | Path,
        audio_path: str | Path,
        thresholds: LipSyncThresholds | None = None,
    ) -> LipSyncQualityEvidence:
        if self.origin not in ALLOWED_ANALYZER_ORIGINS:
            raise LipSyncQCError("unsupported external lip-sync analyzer origin")
        if not self.command:
            raise LipSyncQCError("external lip-sync analyzer command is empty")
        has_video = any("{video}" in item for item in self.command)
        has_audio = any("{audio}" in item for item in self.command)
        if not has_video or not has_audio:
            raise LipSyncQCError(
                "external lip-sync analyzer command must bind {video} and {audio}"
            )
        video = Path(video_path).resolve()
        audio = Path(audio_path).resolve()
        video_sha = _sha256_file(video)
        audio_sha = _sha256_file(audio)
        argv = [
            item.replace("{video}", str(video)).replace("{audio}", str(audio))
            for item in self.command
        ]
        try:
            completed = subprocess.run(
                argv,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise LipSyncQCError("external lip-sync analyzer execution failed") from exc
        try:
            measured = json.loads(completed.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LipSyncQCError(
                "external lip-sync analyzer did not emit valid JSON"
            ) from exc
        if not isinstance(measured, Mapping):
            raise LipSyncQCError("external lip-sync analyzer JSON must be an object")

        limits = (thresholds or LipSyncThresholds()).validated()
        confidence = _unit_interval(
            measured.get("sync_confidence"), field="sync confidence"
        )
        offset = _finite(measured.get("av_offset_ms"), field="A/V offset")
        speaking = _unit_interval(
            measured.get("speaking_frame_coverage"),
            field="speaking-frame coverage",
        )
        face = _unit_interval(
            measured.get("face_track_coverage"), field="face-track coverage"
        )
        accepted = (
            confidence >= limits.min_sync_confidence
            and abs(offset) <= limits.max_abs_offset_ms
            and speaking >= limits.min_speaking_frame_coverage
            and face >= limits.min_face_track_coverage
        )
        evidence = LipSyncQualityEvidence(
            shot_id=_required_text(shot_id, field="shot ID"),
            video_sha256=video_sha,
            audio_sha256=audio_sha,
            analyzer_origin=self.origin,
            analyzer_id=_required_text(
                measured.get("analyzer_id"), field="analyzer ID"
            ),
            analyzer_revision=_required_text(
                measured.get("analyzer_revision"), field="analyzer revision"
            ),
            analyzer_license_id=_required_text(
                measured.get("analyzer_license_id"), field="analyzer license ID"
            ),
            analyzer_source_url=_required_text(
                measured.get("analyzer_source_url"), field="analyzer source URL"
            ),
            sync_confidence=confidence,
            av_offset_ms=offset,
            speaking_frame_coverage=speaking,
            face_track_coverage=face,
            measured=True,
            accepted=accepted,
        )
        return validate_lipsync_quality_evidence(
            evidence.to_dict(),
            expected_shot_id=shot_id,
            expected_video_sha256=video_sha,
            expected_audio_sha256=audio_sha,
            thresholds=limits,
        )
