"""Versioned production runtime identity for durable CINEOS film jobs.

Long-running native film generation must fail closed when a resumed job is bound to
materially different renderer/continuity policy.  This manifest captures the small
set of production invariants that affect durable temporal state and acceptance
semantics, while deliberately excluding incidental process details.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

PRODUCTION_RUNTIME_MANIFEST_SCHEMA = "cineos-production-runtime/0.1"


@dataclass(frozen=True, slots=True)
class ProductionRuntimeManifest:
    """Stable identity for a production FIRST FILM runtime composition."""

    renderer_id: str
    temporal_model_fingerprint: str
    device: str
    max_recovery_attempts: int
    require_final_film_evaluation: bool
    require_audio: bool
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
        if self.schema != PRODUCTION_RUNTIME_MANIFEST_SCHEMA:
            raise ValueError("unsupported production runtime manifest schema")

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe deterministic manifest payload."""
        return asdict(self)

    @classmethod
    def restore(cls, payload: dict[str, object]) -> ProductionRuntimeManifest:
        """Restore a manifest, rejecting unknown schemas or incomplete payloads."""
        if str(payload.get("schema", "")) != PRODUCTION_RUNTIME_MANIFEST_SCHEMA:
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
        return cls(
            renderer_id=str(payload["renderer_id"]),
            temporal_model_fingerprint=str(payload["temporal_model_fingerprint"]),
            device=str(payload["device"]),
            max_recovery_attempts=int(payload["max_recovery_attempts"]),
            require_final_film_evaluation=bool(
                payload["require_final_film_evaluation"]
            ),
            require_audio=bool(payload["require_audio"]),
        )

    def assert_resume_compatible(self, saved: ProductionRuntimeManifest) -> None:
        """Fail closed when a durable job cannot safely resume on this runtime.

        Recovery budget changes are intentionally compatible: they affect how many
        future retries are permitted, not the semantics of already accepted shots.
        Renderer identity, model weights, device policy and final acceptance gates
        are treated as production invariants.
        """
        mismatches: list[str] = []
        for field_name in (
            "renderer_id",
            "temporal_model_fingerprint",
            "device",
            "require_final_film_evaluation",
            "require_audio",
        ):
            if getattr(self, field_name) != getattr(saved, field_name):
                mismatches.append(field_name)
        if mismatches:
            raise ValueError(
                "production runtime is incompatible with saved job: "
                + ", ".join(mismatches)
            )


__all__ = [
    "PRODUCTION_RUNTIME_MANIFEST_SCHEMA",
    "ProductionRuntimeManifest",
]
