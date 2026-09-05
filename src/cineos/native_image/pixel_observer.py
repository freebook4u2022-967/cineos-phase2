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

PIXEL_CONTINUITY_MEMORY_SCHEMA = "cineos-pixel-continuity-memory/0.2"
PIXEL_AWARE_OBSERVER_CHECKPOINT_SCHEMA = "cineos-pixel-aware-observer-checkpoint/0.1"
_LEGACY_PIXEL_CONTINUITY_MEMORY_SCHEMAS = {"cineos-pixel-continuity-memory/0.1"}
_LUMA_BINS = 8
_SPATIAL_CELLS = 4


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
    spatial_luma: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    edge_energy: float = 0.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("pixel descriptor dimensions must be positive")
        if len(self.luma_histogram) != _LUMA_BINS:
            raise ValueError(f"luma histogram must contain {_LUMA_BINS} bins")
        if len(self.spatial_luma) != _SPATIAL_CELLS:
            raise ValueError(f"spatial luma must contain {_SPATIAL_CELLS} cells")
        bounded = (
            self.mean_luma,
            self.mean_red,
            self.mean_green,
            self.mean_blue,
            self.black_fraction,
            self.clipped_fraction,
            self.edge_energy,
            *self.luma_histogram,
            *self.spatial_luma,
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
            "spatial_luma": list(self.spatial_luma),
            "edge_energy": self.edge_energy,
        }

    @classmethod
    def restore(cls, payload: dict[str, object]) -> PixelFrameDescriptor:
        histogram = payload.get("luma_histogram")
        if not isinstance(histogram, list):
            raise ValueError("pixel descriptor snapshot is missing luma histogram")
        raw_spatial = payload.get("spatial_luma", [0.0] * _SPATIAL_CELLS)
        if not isinstance(raw_spatial, list):
            raise ValueError("pixel descriptor spatial luma must be a list")
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
            spatial_luma=tuple(float(value) for value in raw_spatial),
            edge_energy=float(payload.get("edge_energy", 0.0)),
        )


def _pixel_luma(frame: DecodedRGBFrame, x: int, y: int) -> float:
    offset = (y * frame.width + x) * 3
    red = frame.rgb[offset] / 255.0
    green = frame.rgb[offset + 1] / 255.0
    blue = frame.rgb[offset + 2] / 255.0
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def describe_rgb_frame(frame: DecodedRGBFrame) -> PixelFrameDescriptor:
    """Extract deterministic luminance, color, and spatial RGB evidence."""
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
    quadrant_sums = [0.0] * _SPATIAL_CELLS
    quadrant_counts = [0] * _SPATIAL_CELLS
    luma_grid = [0.0] * count

    for y in range(frame.height):
        for x in range(frame.width):
            offset = (y * frame.width + x) * 3
            red = frame.rgb[offset] / 255.0
            green = frame.rgb[offset + 1] / 255.0
            blue = frame.rgb[offset + 2] / 255.0
            luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            luma_grid[y * frame.width + x] = luma

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

            quadrant = (2 if y * 2 >= frame.height else 0) + (
                1 if x * 2 >= frame.width else 0
            )
            quadrant_sums[quadrant] += luma
            quadrant_counts[quadrant] += 1

    edge_total = 0.0
    edge_count = 0
    for y in range(frame.height):
        for x in range(frame.width):
            current = luma_grid[y * frame.width + x]
            if x + 1 < frame.width:
                edge_total += abs(current - luma_grid[y * frame.width + x + 1])
                edge_count += 1
            if y + 1 < frame.height:
                edge_total += abs(current - luma_grid[(y + 1) * frame.width + x])
                edge_count += 1

    mean_luma = luma_total / count
    variance = max(0.0, luma_sq_total / count - mean_luma * mean_luma)
    spatial_luma = tuple(
        (
            quadrant_sums[index] / quadrant_counts[index]
            if quadrant_counts[index]
            else mean_luma
        )
        for index in range(_SPATIAL_CELLS)
    )
    edge_energy = edge_total / edge_count if edge_count else 0.0
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
        spatial_luma=spatial_luma,
        edge_energy=edge_energy,
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
    spatial_delta = (
        sum(
            abs(left - right)
            for left, right in zip(
                baseline.spatial_luma, candidate.spatial_luma, strict=True
            )
        )
        / _SPATIAL_CELLS
    )
    edge_delta = abs(baseline.edge_energy - candidate.edge_energy)
    delta = (
        0.45 * histogram_tv
        + 0.25 * color_delta
        + 0.20 * spatial_delta
        + 0.10 * edge_delta
    )
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
        schema = payload.get("schema")
        if schema != PIXEL_CONTINUITY_MEMORY_SCHEMA and schema not in (
            _LEGACY_PIXEL_CONTINUITY_MEMORY_SCHEMAS
        ):
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

    def checkpoint_state(self) -> dict[str, object]:
        """Capture observer-owned continuity state before an acceptance transaction."""
        return {
            "schema": PIXEL_AWARE_OBSERVER_CHECKPOINT_SCHEMA,
            "pixel_memory": self.pixel_observer.memory.snapshot(),
        }

    def restore_state(self, checkpoint: object) -> None:
        """Restore an earlier checkpoint without replacing the observer instance."""
        if not isinstance(checkpoint, dict):
            raise TypeError("pixel-aware observer checkpoint must be a mapping")
        if checkpoint.get("schema") != PIXEL_AWARE_OBSERVER_CHECKPOINT_SCHEMA:
            raise ValueError("unsupported pixel-aware observer checkpoint schema")
        pixel_memory = checkpoint.get("pixel_memory")
        if not isinstance(pixel_memory, dict):
            raise ValueError("pixel-aware observer checkpoint is missing pixel memory")
        self.pixel_observer.memory = PixelContinuityMemory.restore(pixel_memory)

    def accept_frame(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> None:
        self.pixel_observer.accept(result, plan)
