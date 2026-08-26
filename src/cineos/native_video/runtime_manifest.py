"""Versioned production runtime identity for durable CINEOS film jobs.

Long-running native film generation must fail closed when a resumed job is bound to
materially different renderer/continuity policy. This manifest captures the small
set of production invariants that affect durable temporal state and acceptance
semantics, while also recording operational settings for auditability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PRODUCTION_RUNTIME_MANIFEST_SCHEMA = "cineos-production-runtime/0.1"
LEGACY_UNBOUND_FINAL_GATE_POLICY = "legacy-unbound-final-gate-policy"


@dataclass(frozen=True, slots=True)
class ProductionRuntimeManifest:
    """Stable identity for a production FIRST FILM runtime composition."""

    renderer_id: str
    temporal_model_fingerprint: str
    device: str
    max_recovery_attempts: int
    require_final_film_evaluation: bool
    require_audio: bool
    final_gate_policy_fingerprint: str = LEGACY_UNBOUND_FINAL_GATE_POLICY
    schema: str = PRODUCTION_RUNTIME_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if not self.renderer_id.strip():
            raise ValueError("renderer_id must not be empty")
        if not self.temporal_model_fingerprint.strip():
            raise ValueError("temporal_model_fingerprint must not be empty")
        if not self.device.strip():
            raise ValueError("device must not be empty")
        if self.max_recovery_attempts < 0:
            raise ValueError("max_recovery_attempts must be non-negative")
        if not self.final_gate_policy_fingerprint.strip():
            raise ValueError("final_gate_policy_fingerprint must not be empty")
        if self.schema != PRODUCTION_RUNTIME_MANIFEST_SCHEMA:
            raise ValueError("unsupported production runtime manifest schema")

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe deterministic manifest payload."""
        return asdict(self)

    @classmethod
    def restore(cls, payload: dict[str, object]) -> ProductionRuntimeManifest:
        """Restore a manifest, rejecting unknown schemas or malformed payloads.

        Manifests written before final-gate policy binding are still parseable so
        operators receive a semantic incompatibility error rather than a malformed
        checkpoint error. They intentionally restore with an explicit legacy marker;
        a current production runtime therefore refuses to resume them until the job
        is restarted under a known acceptance policy.
        """
        if payload.get("schema") != PRODUCTION_RUNTIME_MANIFEST_SCHEMA:
            raise ValueError("unsupported production runtime manifest schema")
        required = {
            "renderer_id",
            "temporal_model_fingerprint",
            "device",
            "max_recovery_attempts",
            "require_final_film_evaluation",
            "require_audio",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError(
                "production runtime manifest is missing: " + ", ".join(missing)
            )

        renderer_id = payload["renderer_id"]
        fingerprint = payload["temporal_model_fingerprint"]
        device = payload["device"]
        recovery_budget = payload["max_recovery_attempts"]
        require_final = payload["require_final_film_evaluation"]
        require_audio = payload["require_audio"]
        final_gate_policy = payload.get(
            "final_gate_policy_fingerprint", LEGACY_UNBOUND_FINAL_GATE_POLICY
        )
        if not isinstance(renderer_id, str):
            raise ValueError("production runtime renderer_id must be a string")
        if not isinstance(fingerprint, str):
            raise ValueError(
                "production runtime temporal_model_fingerprint must be a string"
            )
        if not isinstance(device, str):
            raise ValueError("production runtime device must be a string")
        if isinstance(recovery_budget, bool) or not isinstance(recovery_budget, int):
            raise ValueError(
                "production runtime max_recovery_attempts must be an integer"
            )
        if not isinstance(require_final, bool):
            raise ValueError(
                "production runtime require_final_film_evaluation must be boolean"
            )
        if not isinstance(require_audio, bool):
            raise ValueError("production runtime require_audio must be boolean")
        if not isinstance(final_gate_policy, str) or not final_gate_policy.strip():
            raise ValueError(
                "production runtime final_gate_policy_fingerprint must be a "
                "non-empty string"
            )

        return cls(
            renderer_id=renderer_id,
            temporal_model_fingerprint=fingerprint,
            device=device,
            max_recovery_attempts=recovery_budget,
            require_final_film_evaluation=require_final,
            require_audio=require_audio,
            final_gate_policy_fingerprint=final_gate_policy,
        )

    def assert_resume_compatible(self, saved: ProductionRuntimeManifest) -> None:
        """Fail closed when durable production semantics changed across a resume.

        Device placement and recovery-budget changes are intentionally compatible.
        Identical temporal weights can move between CPU/GPU without invalidating
        recurrent state, and retry budget only controls future attempts. Renderer
        identity, model weights, final-gate policy, and final acceptance requirements
        remain invariants.
        """
        mismatches: list[str] = []
        for field_name in (
            "renderer_id",
            "temporal_model_fingerprint",
            "require_final_film_evaluation",
            "require_audio",
            "final_gate_policy_fingerprint",
        ):
            if getattr(self, field_name) != getattr(saved, field_name):
                mismatches.append(field_name)
        if mismatches:
            raise ValueError(
                "production runtime is incompatible with saved job: "
                + ", ".join(mismatches)
            )


__all__ = [
    "LEGACY_UNBOUND_FINAL_GATE_POLICY",
    "PRODUCTION_RUNTIME_MANIFEST_SCHEMA",
    "ProductionRuntimeManifest",
]
