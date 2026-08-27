"""Fail-closed production readiness accounting for native CINEOS V1.

The repository can contain a complete orchestration stack while still lacking the
trained native artifacts or acceptance evidence required to truthfully call a build
production ready. This module makes that distinction machine-readable so release
automation, dashboards, and operators share the same contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .runtime_manifest import (
    LEGACY_UNBOUND_FINAL_GATE_POLICY,
    LEGACY_UNBOUND_NATIVE_MODEL_MANIFEST,
    ProductionRuntimeManifest,
)

READINESS_EVIDENCE_KEYS = (
    "native_model_training",
    "native_model_benchmark",
    "temporal_continuity_benchmark",
    "character_identity_benchmark",
    "audio_dialogue_gate",
    "full_film_e2e",
    "release_audit",
)


@dataclass(frozen=True, slots=True)
class ProductionReadinessEvidence:
    """Evidence required before CINEOS V1 may be declared production ready."""

    runtime_manifest: ProductionRuntimeManifest
    native_model_trained: bool
    native_model_benchmark_passed: bool
    temporal_continuity_benchmark_passed: bool
    character_identity_benchmark_passed: bool
    audio_dialogue_gate_passed: bool
    full_film_e2e_passed: bool
    release_audit_passed: bool
    external_gpu_training_required: bool = False


@dataclass(frozen=True, slots=True)
class ReadinessEvidenceArtifact:
    """One immutable artifact attesting a production-readiness claim."""

    key: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        if self.key not in READINESS_EVIDENCE_KEYS:
            raise ValueError(f"unsupported readiness evidence key: {self.key}")
        if not self.path.strip():
            raise ValueError("readiness evidence path must not be empty")
        normalized = self.sha256.strip().lower()
        if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
            raise ValueError("readiness evidence sha256 must be one SHA-256 hex digest")
        object.__setattr__(self, "sha256", normalized)

    def verify(self) -> str | None:
        """Return a blocker when the artifact is missing or has changed."""
        path = Path(self.path)
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            return f"readiness evidence artifact is missing: {self.key}"
        except OSError as error:
            return f"readiness evidence artifact is unreadable: {self.key}: {error}"
        digest = hashlib.sha256(payload).hexdigest()
        if digest != self.sha256:
            return f"readiness evidence artifact digest mismatch: {self.key}"
        return None


@dataclass(frozen=True, slots=True)
class ProductionReadinessAttestation:
    """Runtime-bound, content-addressed evidence for a production-ready claim."""

    runtime_manifest_fingerprint: str
    artifacts: tuple[ReadinessEvidenceArtifact, ...]

    def __post_init__(self) -> None:
        fingerprint = self.runtime_manifest_fingerprint.strip().lower()
        if len(fingerprint) != 64 or any(
            ch not in "0123456789abcdef" for ch in fingerprint
        ):
            raise ValueError(
                "runtime_manifest_fingerprint must be one SHA-256 hex digest"
            )
        object.__setattr__(self, "runtime_manifest_fingerprint", fingerprint)
        keys = tuple(artifact.key for artifact in self.artifacts)
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise ValueError(
                "duplicate readiness evidence keys: " + ", ".join(duplicates)
            )


@dataclass(frozen=True, slots=True)
class ProductionReadinessReport:
    """Deterministic readiness result with explicit blocking reasons."""

    ready: bool
    blockers: tuple[str, ...]

    def require_ready(self) -> None:
        """Raise with all blockers when a caller attempts a premature release."""
        if self.ready:
            return
        raise RuntimeError(
            "CINEOS V1 is not production ready: " + "; ".join(self.blockers)
        )


def evaluate_production_readiness(
    evidence: ProductionReadinessEvidence,
) -> ProductionReadinessReport:
    """Evaluate the complete V1 release boundary without optimistic fallbacks.

    A production runtime must be bound to a real native model manifest and a real
    final-gate policy. Passing unit/integration tests alone is intentionally
    insufficient: the learned model, visual/temporal/identity benchmarks, audio,
    complete film E2E, and release audit must all have explicit evidence.
    """
    if not isinstance(evidence, ProductionReadinessEvidence):
        raise TypeError("evidence must be ProductionReadinessEvidence")

    blockers: list[str] = []
    runtime = evidence.runtime_manifest

    if runtime.native_model_manifest_sha256 == LEGACY_UNBOUND_NATIVE_MODEL_MANIFEST:
        blockers.append("production runtime is not bound to a native model manifest")
    if runtime.final_gate_policy_fingerprint == LEGACY_UNBOUND_FINAL_GATE_POLICY:
        blockers.append("production runtime is not bound to a final-gate policy")
    if not runtime.require_final_film_evaluation:
        blockers.append("final film evaluation is disabled")
    if not runtime.require_audio:
        blockers.append("production audio acceptance is disabled")

    required_evidence = (
        (evidence.native_model_trained, "native model training is not complete"),
        (
            evidence.native_model_benchmark_passed,
            "native model quality benchmark has not passed",
        ),
        (
            evidence.temporal_continuity_benchmark_passed,
            "temporal continuity benchmark has not passed",
        ),
        (
            evidence.character_identity_benchmark_passed,
            "character identity benchmark has not passed",
        ),
        (
            evidence.audio_dialogue_gate_passed,
            "audio/dialogue acceptance gate has not passed",
        ),
        (
            evidence.full_film_e2e_passed,
            "full-film end-to-end validation has not passed",
        ),
        (evidence.release_audit_passed, "release audit has not passed"),
    )
    blockers.extend(reason for passed, reason in required_evidence if not passed)

    if evidence.external_gpu_training_required:
        blockers.append("external GPU/model training dependency remains")

    return ProductionReadinessReport(ready=not blockers, blockers=tuple(blockers))


def evaluate_attested_production_readiness(
    evidence: ProductionReadinessEvidence,
    attestation: ProductionReadinessAttestation,
) -> ProductionReadinessReport:
    """Evaluate readiness and verify durable proof for every passing claim.

    This stricter release boundary is intended for production automation. The
    original evaluator remains available for compatibility and planning, while a
    real release must bind the exact runtime manifest and content-addressed artifacts
    produced by training, benchmark, audio, E2E, and audit stages.
    """
    if not isinstance(attestation, ProductionReadinessAttestation):
        raise TypeError("attestation must be ProductionReadinessAttestation")

    base = evaluate_production_readiness(evidence)
    blockers = list(base.blockers)
    runtime_fingerprint = evidence.runtime_manifest.fingerprint
    if attestation.runtime_manifest_fingerprint != runtime_fingerprint:
        blockers.append("readiness attestation is bound to a different runtime manifest")

    by_key = {artifact.key: artifact for artifact in attestation.artifacts}
    for key in READINESS_EVIDENCE_KEYS:
        artifact = by_key.get(key)
        if artifact is None:
            blockers.append(f"readiness evidence artifact is missing: {key}")
            continue
        blocker = artifact.verify()
        if blocker is not None:
            blockers.append(blocker)

    return ProductionReadinessReport(ready=not blockers, blockers=tuple(blockers))


__all__ = [
    "READINESS_EVIDENCE_KEYS",
    "ProductionReadinessAttestation",
    "ProductionReadinessEvidence",
    "ProductionReadinessReport",
    "ReadinessEvidenceArtifact",
    "evaluate_attested_production_readiness",
    "evaluate_production_readiness",
]
