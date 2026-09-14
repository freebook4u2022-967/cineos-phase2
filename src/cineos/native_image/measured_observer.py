"""Transactional composition of measured native-frame continuity evidence.

CINEOS has multiple complementary continuity observers: low-level generated-pixel
statistics catch lighting/color drift, while denser spatial evidence catches
composition changes that preserve the same global histogram.  Production frame QC
must consume both without letting either observer advance durable state until the
whole frame has been accepted.

This module provides that composition root.  Semantic identity remains a separately
trained/injected source; pixel statistics are never treated as proof of identity.
Overlapping visual axes are merged conservatively by taking the minimum score so a
strong global statistic cannot hide a measured spatial failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .backend import NativeImageResearchResult
from .conditioning import NativeImageConditioningPlan
from .pixel_observer import (
    DecodedPixelContinuityObserver,
    IdentityObservationSource,
    PixelContinuityMemory,
)
from .spatial_evidence import (
    MeasuredSpatialContinuityObserver,
    SpatialContinuityMemory,
)
from .temporal_identity import IdentityObservation
from .visual_qc import VisualContinuityObservation

MEASURED_NATIVE_FRAME_OBSERVER_SCHEMA = "cineos-measured-native-frame-observer/0.1"


class VisualContinuitySource(Protocol):
    """Measured visual evidence source with explicit acceptance semantics."""

    def observe(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> VisualContinuityObservation: ...

    def accept(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> None: ...


def merge_visual_observations(
    *observations: VisualContinuityObservation,
) -> VisualContinuityObservation:
    """Merge compatible observations without averaging away a failed axis.

    Each source may report a different subset of the provider-neutral visual axes.
    When two measured sources report the same axis, the lower score wins.  This is a
    deliberate fail-closed policy for production continuity QC: independent evidence
    of a failure must survive composition rather than being diluted by averaging.
    Confidence is likewise the minimum confidence across contributing observations.
    """

    if not observations:
        raise ValueError("at least one visual continuity observation is required")
    shot_id = observations[0].shot_id
    if any(observation.shot_id != shot_id for observation in observations[1:]):
        raise ValueError("visual continuity observations must belong to the same shot")

    scores: dict[str, float] = {}
    for observation in observations:
        for axis, score in observation.scores.items():
            existing = scores.get(axis)
            scores[axis] = score if existing is None else min(existing, score)

    return VisualContinuityObservation(
        shot_id=shot_id,
        scores=scores,
        confidence=min(observation.confidence for observation in observations),
    )


@dataclass(slots=True)
class MeasuredNativeFrameObserver:
    """Production observer combining semantic identity with measured RGB evidence.

    ``NativeFrameRuntime`` discovers ``checkpoint_state``/``restore_state`` and
    ``accept_frame`` dynamically.  Implementing all three here makes pixel and
    spatial continuity one transaction: a failed commit restores both stores, and a
    rejected generation never becomes the baseline for the next attempt.
    """

    identity_source: IdentityObservationSource
    pixel_observer: DecodedPixelContinuityObserver = field(
        default_factory=DecodedPixelContinuityObserver
    )
    spatial_observer: MeasuredSpatialContinuityObserver = field(
        default_factory=MeasuredSpatialContinuityObserver
    )

    def observe_identity(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> tuple[IdentityObservation, ...]:
        return self.identity_source.observe_identity(result, plan)

    def observe_visual_continuity(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> VisualContinuityObservation:
        return merge_visual_observations(
            self.pixel_observer.observe(result, plan),
            self.spatial_observer.observe(result, plan),
        )

    def checkpoint_state(self) -> dict[str, object]:
        """Return versioned JSON-safe state for atomic frame acceptance."""

        return {
            "schema": MEASURED_NATIVE_FRAME_OBSERVER_SCHEMA,
            "pixel": self.pixel_observer.memory.snapshot(),
            "spatial": self.spatial_observer.memory.snapshot(),
        }

    def restore_state(self, payload: object) -> None:
        """Restore both measured continuity stores from one validated checkpoint."""

        if not isinstance(payload, dict):
            raise ValueError("measured observer checkpoint must be a mapping")
        if payload.get("schema") != MEASURED_NATIVE_FRAME_OBSERVER_SCHEMA:
            raise ValueError("unsupported measured observer checkpoint schema")
        raw_pixel = payload.get("pixel")
        raw_spatial = payload.get("spatial")
        if not isinstance(raw_pixel, dict) or not isinstance(raw_spatial, dict):
            raise ValueError("measured observer checkpoint is incomplete")

        pixel = PixelContinuityMemory.restore(raw_pixel)
        spatial = SpatialContinuityMemory.restore(raw_spatial)
        # Assign only after both payloads validate, preventing half-restored state.
        self.pixel_observer.memory = pixel
        self.spatial_observer.memory = spatial

    def accept_frame(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> None:
        """Commit all measured evidence atomically after frame-level QC acceptance."""

        checkpoint = self.checkpoint_state()
        try:
            self.pixel_observer.accept(result, plan)
            self.spatial_observer.accept(result, plan)
        except Exception:
            self.restore_state(checkpoint)
            raise


__all__ = [
    "MEASURED_NATIVE_FRAME_OBSERVER_SCHEMA",
    "MeasuredNativeFrameObserver",
    "VisualContinuitySource",
    "merge_visual_observations",
]
