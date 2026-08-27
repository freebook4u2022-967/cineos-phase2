"""Fail-closed production readiness accounting for native CINEOS V1.

The repository can contain a complete orchestration stack while still lacking the
trained native artifacts or acceptance evidence required to truthfully call a build
production ready. This module makes that distinction machine-readable so release
automation, dashboards, and operators share the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass

from .runtime_manifest import (
    LEGACY_UNBOUND_FINAL_GATE_POLICY,
    LEGACY_UNBOUND_NATIVE_MODEL_MANIFEST,
    ProductionRuntimeManifest,
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


__all__ = [
    "ProductionReadinessEvidence",
    "ProductionReadinessReport",
    "evaluate_production_readiness",
]
