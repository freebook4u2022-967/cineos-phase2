"""Quality-gated native model release and rollback orchestration.

This module joins checkpoint benchmarking with the versioned native-model registry so
CINEOS can evolve learned artifacts without activating a merely newer checkpoint.
A candidate must satisfy both runtime compatibility and measurable quality improvement
before it can become active. Rejected candidates leave the active registry untouched.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .checkpoint_gate import (
    CheckpointBenchmarkGate,
    CheckpointPromotionDecision,
    CheckpointScore,
)
from .model_manifest import (
    ModelManifestError,
    NativeModelManifest,
    NativeModelRegistry,
    check_runtime_compatibility,
)

MODEL_RELEASE_RECORD_SCHEMA = "cineos-native-model-release/1"


@dataclass(frozen=True, slots=True)
class NativeModelReleaseDecision:
    """Auditable decision for one attempted native-model activation."""

    promoted: bool
    manifest_sha256: str
    model_id: str
    model_version: str
    quality: CheckpointPromotionDecision
    reason: str
    previous_manifest_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": MODEL_RELEASE_RECORD_SCHEMA,
            "promoted": self.promoted,
            "manifest_sha256": self.manifest_sha256,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "quality": asdict(self.quality),
            "reason": self.reason,
            "previous_manifest_sha256": self.previous_manifest_sha256,
        }


@dataclass(slots=True)
class NativeModelReleaseController:
    """Promote only compatible checkpoints that beat the incumbent benchmark.

    The release controller intentionally keeps quality evaluation separate from model
    registration. Compatibility and benchmark checks happen before registry mutation,
    so a rejected candidate cannot displace the currently active production model.
    """

    registry: NativeModelRegistry
    benchmark_gate: CheckpointBenchmarkGate
    release_record_path: Path | None = None

    def _compatibility_reason(self, manifest: NativeModelManifest) -> str | None:
        compatibility = check_runtime_compatibility(
            manifest,
            runtime_contract_version=self.registry.runtime_contract_version,
            supported_component_contracts=self.registry.supported_component_contracts,
        )
        return None if compatibility.compatible else compatibility.reason

    def _write_record(self, decision: NativeModelReleaseDecision) -> None:
        if self.release_record_path is None:
            return
        destination = Path(self.release_record_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(decision.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    def evaluate(
        self,
        manifest: NativeModelManifest,
        candidate_score: CheckpointScore,
        incumbent_score: CheckpointScore | None,
    ) -> NativeModelReleaseDecision:
        manifest.validate()
        incompatible_reason = self._compatibility_reason(manifest)
        quality = self.benchmark_gate.evaluate(candidate_score, incumbent_score)
        previous = self.registry.active()
        previous_digest = previous.manifest_sha256 if previous is not None else None

        if incompatible_reason is not None:
            return NativeModelReleaseDecision(
                promoted=False,
                manifest_sha256=manifest.manifest_sha256,
                model_id=manifest.model_id,
                model_version=manifest.model_version,
                quality=quality,
                reason="runtime compatibility rejected candidate: "
                + incompatible_reason,
                previous_manifest_sha256=previous_digest,
            )
        if not quality.promoted:
            return NativeModelReleaseDecision(
                promoted=False,
                manifest_sha256=manifest.manifest_sha256,
                model_id=manifest.model_id,
                model_version=manifest.model_version,
                quality=quality,
                reason="benchmark gate rejected candidate: " + quality.reason,
                previous_manifest_sha256=previous_digest,
            )
        return NativeModelReleaseDecision(
            promoted=True,
            manifest_sha256=manifest.manifest_sha256,
            model_id=manifest.model_id,
            model_version=manifest.model_version,
            quality=quality,
            reason="candidate passed runtime compatibility and benchmark gates",
            previous_manifest_sha256=previous_digest,
        )

    def promote(
        self,
        manifest: NativeModelManifest,
        candidate_score: CheckpointScore,
        incumbent_score: CheckpointScore | None,
    ) -> NativeModelReleaseDecision:
        """Activate an eligible candidate and persist an auditable release decision."""

        decision = self.evaluate(manifest, candidate_score, incumbent_score)
        if not decision.promoted:
            self._write_record(decision)
            return decision

        try:
            digest = self.registry.activate(manifest)
        except ModelManifestError:
            raise
        if digest != decision.manifest_sha256:
            raise ModelManifestError("registry activated an unexpected model manifest")
        self._write_record(decision)
        return decision

    def rollback(self) -> NativeModelManifest:
        """Rollback through the registry's compatibility-gated history."""

        return self.registry.rollback()


__all__ = [
    "MODEL_RELEASE_RECORD_SCHEMA",
    "NativeModelReleaseController",
    "NativeModelReleaseDecision",
]
