"""Hardened native FIRST FILM composition for production CINEOS runs.

The lower-level film runner intentionally keeps quality gates optional for backwards
compatibility with legacy and lightweight test integrations. Production native video
must not rely on callers remembering each safety-critical option independently.
This module provides one explicit composition root that binds:

* the native temporal renderer to transactional continuity state,
* continuity checkpoint/retry hooks into ``FilmOrchestrator``,
* measured final-film picture/duration/edit QC, and
* measured encoded-audio QC as a required acceptance criterion.

No external video generator is introduced here. FFmpeg/ffprobe remain inspectors and
container tools only through the existing gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cineos.film.first_film import FirstFilmRunner

from .film_bridge import NativeFilmContinuityBridge
from .final_gate import MeasuredFinalFilmGate
from .renderer_binding import NativeFilmRendererBinding, NativeTemporalShotRenderer


@dataclass(frozen=True, slots=True)
class ProductionFirstFilmRuntime:
    """Owned production composition with inspectable native dependencies.

    Keeping the continuity bridge and renderer binding alongside the public runner
    prevents their lifecycle from becoming implicit. Long-running callers may use
    ``continuity`` for diagnostics/checkpoint inspection while invoking ``runner``
    as the provider-neutral complete-film API.
    """

    runner: FirstFilmRunner
    continuity: NativeFilmContinuityBridge
    renderer_binding: NativeFilmRendererBinding
    final_gate: MeasuredFinalFilmGate


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
    """

    if max_recovery_attempts < 0:
        raise ValueError("max_recovery_attempts must be non-negative")

    active_continuity = continuity or NativeFilmContinuityBridge.default(device=device)
    if continuity is not None and continuity.device != device:
        raise ValueError(
            "explicit continuity device does not match requested production device"
        )

    active_gate = final_gate or MeasuredFinalFilmGate(require_audio=True)
    binding = NativeFilmRendererBinding(native_renderer, active_continuity)
    runner = FirstFilmRunner(
        binding,
        validator,
        renderer_id=renderer_id,
        max_recovery_attempts=max_recovery_attempts,
        orchestrator_kwargs=active_continuity.orchestrator_kwargs(),
        final_film_evaluator=active_gate,
        require_final_film_evaluation=True,
    )
    return ProductionFirstFilmRuntime(
        runner=runner,
        continuity=active_continuity,
        renderer_binding=binding,
        final_gate=active_gate,
    )


__all__ = [
    "ProductionFirstFilmRuntime",
    "build_production_first_film_runtime",
]
