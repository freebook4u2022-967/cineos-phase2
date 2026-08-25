"""Hardened native FIRST FILM composition for production CINEOS runs.

The lower-level film runner intentionally keeps quality gates optional for backwards
compatibility with legacy and lightweight test integrations. Production native video
must not rely on callers remembering each safety-critical option independently.
This module provides one explicit composition root that binds:

* the native temporal renderer to transactional continuity state,
* continuity checkpoint/retry hooks into ``FilmOrchestrator``,
* measured final-film picture/duration/edit QC,
* measured encoded-audio QC as a required acceptance criterion, and
* a versioned runtime manifest that is persisted with continuity state and checked
  before any long-running production job is resumed.

No external video generator is introduced here. FFmpeg/ffprobe remain inspectors and
container tools only through the existing gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cineos.film.first_film import FirstFilmRunner

from .film_bridge import NativeFilmContinuityBridge, temporal_model_fingerprint
from .final_gate import MeasuredFinalFilmGate
from .renderer_binding import NativeFilmRendererBinding, NativeTemporalShotRenderer
from .runtime_manifest import ProductionRuntimeManifest

PRODUCTION_FIRST_FILM_RUNTIME_KIND = "cineos-production-first-film-runtime/0.1"


@dataclass(frozen=True, slots=True)
class ProductionFirstFilmRuntime:
    """Owned production composition with inspectable native dependencies.

    Keeping the continuity bridge and renderer binding alongside the public runner
    prevents their lifecycle from becoming implicit. Long-running callers may use
    ``continuity`` for diagnostics/checkpoint inspection while invoking ``runner``
    as the provider-neutral complete-film API. ``manifest`` records the production
    invariants required to decide whether a persisted job can safely resume after a
    software or model upgrade.
    """

    runner: FirstFilmRunner
    continuity: NativeFilmContinuityBridge
    renderer_binding: NativeFilmRendererBinding
    final_gate: MeasuredFinalFilmGate
    manifest: ProductionRuntimeManifest


def _production_checkpoint_hooks(
    continuity: NativeFilmContinuityBridge,
    manifest: ProductionRuntimeManifest,
) -> dict[str, Any]:
    """Bind durable continuity state to the production runtime identity.

    ``FilmOrchestrator`` already integrity-hashes the complete runtime-state object.
    This envelope adds semantic compatibility: even an intact checkpoint is unsafe
    to resume when it was created by a different renderer/model or acceptance
    policy. Operational changes such as CPU/GPU placement and retry budget remain
    compatible according to :meth:`ProductionRuntimeManifest.assert_resume_compatible`.
    """

    def snapshot() -> dict[str, object]:
        return {
            "kind": PRODUCTION_FIRST_FILM_RUNTIME_KIND,
            "runtime_manifest": manifest.snapshot(),
            "continuity": continuity.snapshot(),
        }

    def restore(payload: dict[str, Any]) -> None:
        if str(payload.get("kind", "")) != PRODUCTION_FIRST_FILM_RUNTIME_KIND:
            raise ValueError("unsupported production FIRST FILM runtime checkpoint")
        raw_manifest = payload.get("runtime_manifest")
        if not isinstance(raw_manifest, dict):
            raise ValueError("production runtime checkpoint is missing runtime_manifest")
        raw_continuity = payload.get("continuity")
        if not isinstance(raw_continuity, dict):
            raise ValueError("production runtime checkpoint is missing continuity")

        saved_manifest = ProductionRuntimeManifest.restore(raw_manifest)
        manifest.assert_resume_compatible(saved_manifest)
        # Restore model state only after all production-level invariants pass. This
        # ordering guarantees an incompatible resume cannot mutate live continuity.
        continuity.restore(raw_continuity)

    return {
        "checkpoint_state_provider": snapshot,
        "checkpoint_state_restorer": restore,
        "checkpoint_state_resetter": continuity.reset,
        "shot_attempt_start": continuity.start_attempt,
        "shot_attempt_accepted": continuity.accept_attempt,
        "shot_attempt_rejected": continuity.reject_attempt,
    }


def build_production_first_film_runtime(
    native_renderer: NativeTemporalShotRenderer,
    validator: Any | None = None,
    *,
    continuity: NativeFilmContinuityBridge | None = None,
    final_gate: MeasuredFinalFilmGate | None = None,
    renderer_id: str = "cineos-native",
    max_recovery_attempts: int = 2,
    device: str = "cpu",
) -> ProductionFirstFilmRuntime:
    """Build the fail-closed native CINEOS FIRST FILM production path.

    Production acceptance always requires post-assembly measured QC. When no gate
    is injected, the default gate also requires a real encoded audio stream. A
    custom gate is permitted for policy evolution/testing, but production callers
    cannot accidentally disable *final-film* evaluation itself because the runner
    is always constructed with ``require_final_film_evaluation=True``.

    Durable production checkpoints contain both continuity memory and this runtime's
    versioned manifest. Resume therefore fails before recurrent state is restored if
    renderer identity, temporal weights, or final acceptance requirements changed.
    """

    if max_recovery_attempts < 0:
        raise ValueError("max_recovery_attempts must be non-negative")
    if not renderer_id.strip():
        raise ValueError("renderer_id must not be empty")
    if not device.strip():
        raise ValueError("device must not be empty")

    active_continuity = continuity or NativeFilmContinuityBridge.default(device=device)
    if continuity is not None and continuity.device != device:
        raise ValueError(
            "explicit continuity device does not match requested production device"
        )

    active_gate = final_gate or MeasuredFinalFilmGate(require_audio=True)
    binding = NativeFilmRendererBinding(native_renderer, active_continuity)
    manifest = ProductionRuntimeManifest(
        renderer_id=renderer_id,
        temporal_model_fingerprint=temporal_model_fingerprint(active_continuity.model),
        device=device,
        max_recovery_attempts=max_recovery_attempts,
        require_final_film_evaluation=True,
        require_audio=active_gate.require_audio,
    )
    runner = FirstFilmRunner(
        binding,
        validator,
        renderer_id=renderer_id,
        max_recovery_attempts=max_recovery_attempts,
        orchestrator_kwargs=_production_checkpoint_hooks(active_continuity, manifest),
        final_film_evaluator=active_gate,
        require_final_film_evaluation=True,
    )
    return ProductionFirstFilmRuntime(
        runner=runner,
        continuity=active_continuity,
        renderer_binding=binding,
        final_gate=active_gate,
        manifest=manifest,
    )


__all__ = [
    "PRODUCTION_FIRST_FILM_RUNTIME_KIND",
    "ProductionFirstFilmRuntime",
    "build_production_first_film_runtime",
]
