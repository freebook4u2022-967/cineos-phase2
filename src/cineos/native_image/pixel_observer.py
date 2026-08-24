"""Transactional visual observations derived from generated RGB pixels.

This module closes the gap between native generation and continuity QC without
pretending that low-level pixel statistics solve semantic identity. It extracts
repeatable lighting/environment evidence from CINEOS-owned decoded RGB frames and
only advances its baseline after the enclosing frame runtime accepts a candidate.
Semantic character identity remains an injected, separately trainable concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .backend import NativeImageResearchResult
from .conditioning import NativeImageConditioningPlan
from .neural_decoder import DecodedRGBFrame
from .temporal_identity import IdentityObservation
from .visual_qc import VisualContinuityObservation

PIXEL_CONTINUITY_MEMORY_SCHEMA = "cineos-pixel-continuity-memory/0.1"
_LUMA_BINS = 8


@dataclass(frozen=True, slots=True)
class PixelFrameDescriptor:
    """Compact, JSON-safe evidence extracted directly from one RGB frame."""

    width: int
    height: int
    mean_luma: float
    luma_std: float
    mean_red: float
    mean_green: float
    mean_blue: float
    luma_histogram: tuple[float, ...]
    black_fraction: float
    clipped_fraction: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("pixel descriptor dimensions must be positive")
        if len(self.luma_histogram) != _LUMA_BINS:
            raise ValueError(f"luma histogram must contain {_LUMA_BINS} bins")
        bounded = (
            self.mean_luma,
            self.mean_red,
            self.mean_green,
            self.mean_blue,
            self.black_fraction,
            self.clipped_fraction,
            *self.luma_histogram,
        )
        if any(not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("normalized pixel descriptor values must be in [0, 1]")
        if not 0.0 <= self.luma_std <= 0.5:
            raise ValueError("luma standard deviation must be in [0, 0.5]")

    def snapshot(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "mean_luma": self.mean_luma,
            "luma_std": self.luma_std,
            "mean_red": self.mean_red,
            "mean_green": self.mean_green,
            "mean_blue": self.mean_blue,
            "luma_histogram": list(self.luma_histogram),
            "black_fraction": self.black_fraction,
            "clipped_fraction": self.clipped_fraction,
        }

    @classmethod
    def restore(cls, payload: dict[str, object]) -> PixelFrameDescriptor:
        histogram = payload.get("luma_histogram")
        if not isinstance(histogram, list):
            raise ValueError("pixel descriptor snapshot is missing luma histogram")
        return cls(
            width=int(payload["width"]),
            height=int(payload["height"]),
            mean_luma=float(payload["mean_luma"]),
            luma_std=float(payload["luma_std"]),
            mean_red=float(payload["mean_red"]),
            mean_green=float(payload["mean_green"]),
            mean_blue=float(payload["mean_blue"]),
            luma_histogram=tuple(float(value) for value in histogram),
            black_fraction=float(payload["black_fraction"]),
            clipped_fraction=float(payload["clipped_fraction"]),
        )


def describe_rgb_frame(frame: DecodedRGBFrame) -> PixelFrameDescriptor:
    """Extract deterministic luminance/color statistics from generated RGB bytes."""
    if not isinstance(frame, DecodedRGBFrame):
        raise TypeError("frame must be a DecodedRGBFrame")
    expected = frame.width * frame.height * 3
    if len(frame.rgb) != expected:
        raise ValueError(f"RGB frame expected {expected} bytes, got {len(frame.rgb)}")

    count = frame.width * frame.height
    red_total = green_total = blue_total = 0.0
    luma_total = luma_sq_total = 0.0
    black = clipped = 0
    histogram = [0] * _LUMA_BINS

    for offset in range(0, len(frame.rgb), 3):
        red = frame.rgb[offset] / 255.0
        green = frame.rgb[offset + 1] / 255.0
        blue = frame.rgb[offset + 2] / 255.0
        luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue

        red_total += red
        green_total += green
        blue_total += blue
        luma_total += luma
        luma_sq_total += luma * luma
        histogram[min(_LUMA_BINS - 1, int(luma * _LUMA_BINS))] += 1
        if luma <= 4.0 / 255.0:
            black += 1
        if red >= 251.0 / 255.0 or green >= 251.0 / 255.0 or blue >= 251.0 / 255.0:
            clipped += 1

    mean_luma = luma_total / count
    variance = max(0.0, luma_sq_total / count - mean_luma * mean_luma)
    return PixelFrameDescriptor(
        width=frame.width,
        height=frame.height,
        mean_luma=mean_luma,
        luma_std=variance**0.5,
        mean_red=red_total / count,
        mean_green=green_total / count,
        mean_blue=blue_total / count,
        luma_histogram=tuple(value / count for value in histogram),
        black_fraction=black / count,
        clipped_fraction=clipped / count,
    )


def _environment_similarity(
    baseline: PixelFrameDescriptor, candidate: PixelFrameDescriptor
) -> float:
    histogram_tv = 0.5 * sum(
        abs(left - right)
        for left, right in zip(
            baseline.luma_histogram, candidate.luma_histogram, strict=True
        )
    )
    color_delta = (
        abs(baseline.mean_red - candidate.mean_red)
        + abs(baseline.mean_green - candidate.mean_green)
        + abs(baseline.mean_blue - candidate.mean_blue)
    ) / 3.0
    delta = 0.65 * histogram_tv + 0.35 * color_delta
    return max(0.0, min(1.0, 1.0 - delta))


def _lighting_similarity(
    baseline: PixelFrameDescriptor, candidate: PixelFrameDescriptor
) -> float:
    delta = abs(baseline.mean_luma - candidate.mean_luma) + 2.0 * abs(
        baseline.luma_std - candidate.luma_std
    )
    return max(0.0, min(1.0, 1.0 - delta))


@dataclass(slots=True)
class PixelContinuityMemory:
    """Last accepted generated-pixel baseline per scene."""

    _accepted: dict[str, PixelFrameDescriptor] = field(default_factory=dict)

    def latest(self, scene_id: str) -> PixelFrameDescriptor | None:
        return self._accepted.get(scene_id)

    def accept(self, scene_id: str, descriptor: PixelFrameDescriptor) -> None:
        if not scene_id:
            raise ValueError("pixel continuity memory requires a scene ID")
        self._accepted[scene_id] = descriptor

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": PIXEL_CONTINUITY_MEMORY_SCHEMA,
            "accepted": {
                scene_id: descriptor.snapshot()
                for scene_id, descriptor in sorted(self._accepted.items())
            },
        }

    @classmethod
    def restore(cls, payload: dict[str, object]) -> PixelContinuityMemory:
        if payload.get("schema") != PIXEL_CONTINUITY_MEMORY_SCHEMA:
            raise ValueError("unsupported pixel continuity memory schema")
        raw = payload.get("accepted", {})
        if not isinstance(raw, dict):
            raise ValueError("pixel continuity accepted state must be a mapping")
        memory = cls()
        for scene_id, descriptor_payload in raw.items():
            if not isinstance(scene_id, str) or not isinstance(
                descriptor_payload, dict
            ):
                raise ValueError("invalid pixel continuity memory entry")
            memory.accept(scene_id, PixelFrameDescriptor.restore(descriptor_payload))
        return memory


@dataclass(slots=True)
class DecodedPixelContinuityObserver:
    """Pure observation + explicit commit for generated RGB continuity evidence."""

    memory: PixelContinuityMemory = field(default_factory=PixelContinuityMemory)

    def observe(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> VisualContinuityObservation:
        if result.shot_id != plan.shot_id:
            raise ValueError(
                "generated result and conditioning plan shot IDs must match"
            )
        descriptor = describe_rgb_frame(result.image)
        baseline = self.memory.latest(plan.scene_id)
        if baseline is None:
            environment = lighting = 1.0
        else:
            environment = _environment_similarity(baseline, descriptor)
            lighting = _lighting_similarity(baseline, descriptor)
        return VisualContinuityObservation(
            shot_id=plan.shot_id,
            scores={"environment": environment, "lighting": lighting},
            confidence=1.0,
        )

    def accept(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> None:
        """Commit pixels only after the enclosing QC transaction accepts the frame."""
        if result.shot_id != plan.shot_id:
            raise ValueError(
                "generated result and conditioning plan shot IDs must match"
            )
        self.memory.accept(plan.scene_id, describe_rgb_frame(result.image))


class IdentityObservationSource(Protocol):
    """Separate semantic identity extractor; never inferred from pixel statistics."""

    def observe_identity(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> tuple[IdentityObservation, ...]: ...


@dataclass(slots=True)
class PixelAwareNativeFrameObserver:
    """Compose a real identity extractor with transactional generated-pixel QC."""

    identity_source: IdentityObservationSource
    pixel_observer: DecodedPixelContinuityObserver = field(
        default_factory=DecodedPixelContinuityObserver
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
        return self.pixel_observer.observe(result, plan)

    def accept_frame(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> None:
        self.pixel_observer.accept(result, plan)
