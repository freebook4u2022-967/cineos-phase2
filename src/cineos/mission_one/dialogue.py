"""Dialogue timing and fail-closed lip-sync evidence contracts.

CINEOS owns the policy and evidence boundary defined here.  A lip-sync verifier may
be an external pretrained foundation, but its provenance must be declared explicitly;
this module never represents third-party measurement capability as CINEOS-native.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class LipSyncPolicy:
    """Acceptance thresholds for measured audiovisual speech alignment."""

    max_abs_offset_ms: float = 100.0
    min_confidence: float = 0.80
    min_speech_coverage: float = 0.60

    def __post_init__(self) -> None:
        if not isfinite(self.max_abs_offset_ms) or self.max_abs_offset_ms <= 0:
            raise ValueError("max_abs_offset_ms must be finite and positive")
        for name, value in (
            ("min_confidence", self.min_confidence),
            ("min_speech_coverage", self.min_speech_coverage),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")


@dataclass(frozen=True, slots=True)
class LipSyncEvidence:
    """Artifact-bound measurement emitted by a real lip-sync verifier."""

    verifier_name: str
    verifier_version: str
    verifier_origin: str
    model_id: str
    model_license: str
    audio_sha256: str
    video_sha256: str
    offset_ms: float
    confidence: float
    speech_coverage: float

    def __post_init__(self) -> None:
        for name in (
            "verifier_name",
            "verifier_version",
            "verifier_origin",
            "model_id",
            "model_license",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.verifier_origin not in {
            "external_pretrained_foundation",
            "cineos_native",
        }:
            raise ValueError(
                "verifier_origin must declare native or external provenance"
            )
        for name in ("audio_sha256", "video_sha256"):
            value = getattr(self, name)
            if (
                len(value) != 64
                or value.lower() != value
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not isfinite(self.offset_ms):
            raise ValueError("offset_ms must be finite")
        for name in ("confidence", "speech_coverage"):
            value = getattr(self, name)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")


def evaluate_lip_sync(
    evidence: LipSyncEvidence,
    policy: LipSyncPolicy | None = None,
) -> dict[str, object]:
    """Evaluate measured evidence without relabeling its model provenance."""

    active_policy = policy or LipSyncPolicy()
    reasons: list[str] = []
    if abs(evidence.offset_ms) > active_policy.max_abs_offset_ms:
        reasons.append("av_offset_exceeds_limit")
    if evidence.confidence < active_policy.min_confidence:
        reasons.append("verifier_confidence_below_threshold")
    if evidence.speech_coverage < active_policy.min_speech_coverage:
        reasons.append("speech_coverage_below_threshold")

    return {
        "schema": "cineos-lip-sync-evidence/0.1",
        "status": "measured_pass" if not reasons else "measured_fail",
        "passed": not reasons,
        "reasons": tuple(reasons),
        "policy": asdict(active_policy),
        "evidence": asdict(evidence),
    }


def require_lip_sync_evidence(
    evidence: LipSyncEvidence | None,
    policy: LipSyncPolicy | None = None,
) -> dict[str, object]:
    """Fail closed when production dialogue lacks passing measured evidence."""

    if evidence is None:
        raise ValueError("production dialogue requires measured lip-sync evidence")
    result = evaluate_lip_sync(evidence, policy)
    if not result["passed"]:
        reasons = ", ".join(result["reasons"])
        raise ValueError(f"production lip-sync evidence failed: {reasons}")
    return result


def subtitle_entry(
    shot_id: str,
    text: str,
    start: float,
    duration: float,
    *,
    lip_sync_evidence: LipSyncEvidence | None = None,
    require_measured_lip_sync: bool = False,
    lip_sync_policy: LipSyncPolicy | None = None,
) -> dict:
    """Build a subtitle entry while preserving the legacy unmeasured default."""

    result: dict[str, object] = {
        "shot_id": shot_id,
        "text": text,
        "start": start,
        "end": start + duration,
        "lip_sync": "approximate_unless_measured",
    }
    if lip_sync_evidence is not None:
        measurement = evaluate_lip_sync(lip_sync_evidence, lip_sync_policy)
        result["lip_sync"] = measurement["status"]
        result["lip_sync_evidence"] = measurement
        if require_measured_lip_sync and not measurement["passed"]:
            reasons = ", ".join(measurement["reasons"])
            raise ValueError(f"production lip-sync evidence failed: {reasons}")
    elif require_measured_lip_sync:
        require_lip_sync_evidence(None, lip_sync_policy)
    return result


__all__ = [
    "LipSyncEvidence",
    "LipSyncPolicy",
    "evaluate_lip_sync",
    "require_lip_sync_evidence",
    "subtitle_entry",
]
