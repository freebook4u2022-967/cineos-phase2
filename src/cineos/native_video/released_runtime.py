"""Strict production composition for released CINEOS native video models.

The lower-level production FIRST FILM builder intentionally retains a compatibility
escape hatch for silent legacy/test films. A released native model must not inherit
that escape hatch: release acceptance requires measured encoded audio in addition to
picture, duration, continuity, artifact-integrity, and edit-boundary evidence.

This module provides the release-facing composition root. It delegates model-registry
compatibility and durable manifest binding to ``production_first_film`` while failing
closed if a caller attempts to weaken final-film audio acceptance.
"""

from __future__ import annotations

from typing import Any

from cineos.native_image.model_manifest import NativeModelRegistry

from .final_gate import MeasuredFinalFilmGate
from .production_first_film import (
    ProductionFirstFilmRuntime,
    build_released_production_first_film_runtime,
)
from .renderer_binding import NativeTemporalShotRenderer


def build_strict_released_production_runtime(
    native_renderer: NativeTemporalShotRenderer,
    model_registry: NativeModelRegistry,
    validator: Any | None = None,
    **kwargs: Any,
) -> ProductionFirstFilmRuntime:
    """Build a released-model runtime with non-bypassable measured audio QC.

    ``build_released_production_first_film_runtime`` remains backward compatible and
    therefore accepts a custom ``MeasuredFinalFilmGate(require_audio=False)``. That is
    useful for legacy integration tests but is unsafe as a release entrypoint. This
    wrapper makes the release invariant explicit and auditable.

    Callers may still provide a custom gate to evolve thresholds/evaluators, but it
    must require encoded-audio integrity. If no gate is supplied, a strict measured
    gate is installed automatically.
    """

    requested_gate = kwargs.get("final_gate")
    if requested_gate is None:
        kwargs["final_gate"] = MeasuredFinalFilmGate(require_audio=True)
    elif not isinstance(requested_gate, MeasuredFinalFilmGate):
        raise TypeError("released production final_gate must be MeasuredFinalFilmGate")
    elif not requested_gate.require_audio:
        raise ValueError(
            "released production runtime requires measured final-film audio QC"
        )

    runtime = build_released_production_first_film_runtime(
        native_renderer,
        model_registry,
        validator,
        **kwargs,
    )
    if not runtime.final_gate.require_audio or not runtime.manifest.require_audio:
        raise RuntimeError(
            "released production runtime was composed without required audio QC"
        )
    return runtime


__all__ = ["build_strict_released_production_runtime"]
