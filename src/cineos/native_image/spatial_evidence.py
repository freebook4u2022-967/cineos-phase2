"""Measured spatial continuity evidence derived from CINEOS-owned RGB frames.

Global luminance and color statistics are useful but cannot detect important
continuity failures such as a bright subject moving to the wrong side of frame,
a large foreground object disappearing, or a composition becoming spatially
flat while preserving the same average brightness.  This module adds a small,
dependency-free spatial descriptor that can be evaluated in CI and production
without pretending to solve semantic character identity.

The descriptor is intentionally deterministic and JSON-safe.  It uses a 4x4
luminance grid plus horizontal/vertical edge energy measured from decoded RGB
bytes.  A transactional memory stores only accepted scene baselines, matching
the existing CINEOS rule that rejected generations must never poison durable
continuity state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backend import NativeImageResearchResult
from .conditioning import NativeImageConditioningPlan
from .neural_decoder import DecodedRGBFrame
from .visual_qc import VisualContinuityObservation

SPATIAL_FRAME_EVIDENCE_SCHEMA = "cineos-spatial-frame-evidence/0.1"
_GRID_SIZE = 4


def _luma(red: int, green: int, blue: int) -> float:
    return (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0


@dataclass(frozen=True, slots=True)
class SpatialFrameDescriptor:
    """Compact spatial evidence extracted from one decoded RGB frame."""

    width: int
    height: int
    luma_grid: tuple[float, ...]
    horizontal_edge_energy: float
    vertical_edge_energy: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("spatial descriptor dimensions must be positive")
        if len(self.luma_grid) != _GRID_SIZE * _GRID_SIZE:
            raise ValueError("spatial descriptor requires a 4x4 luma grid")
        values = (
            *self.luma_grid,
            self.horizontal_edge_energy,
            self.vertical_edge_energy,
        )
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("spatial descriptor values must be normalized to [0, 1]")

    def snapshot(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "luma_grid": list(self.luma_grid),
            "horizontal_edge_energy": self.horizontal_edge_energy,
            "vertical_edge_energy": self.vertical_edge_energy,
        }

    @classmethod
    def restore(cls, payload: dict[str, object]) -> SpatialFrameDescriptor:
        grid = payload.get("luma_grid")
        if not isinstance(grid, list):
            raise ValueError("spatial descriptor snapshot is missing luma_grid")
        return cls(
            width=int(payload["width"]),
            height=int(payload["height"]),
            luma_grid=tuple(float(value) for value in grid),
            horizontal_edge_energy=float(payload["horizontal_edge_energy"]),
            vertical_edge_energy=float(payload["vertical_edge_energy"]),
        )


def describe_spatial_rgb_frame(frame: DecodedRGBFrame) -> SpatialFrameDescriptor:
    """Measure coarse composition and edge structure from decoded RGB bytes."""

    if not isinstance(frame, DecodedRGBFrame):
        raise TypeError("frame must be a DecodedRGBFrame")
    expected = frame.width * frame.height * 3
    if len(frame.rgb) != expected:
        raise ValueError(f"RGB frame expected {expected} bytes, got {len(frame.rgb)}")

    lumas = [0.0] * (frame.width * frame.height)
    grid_sums = [0.0] * (_GRID_SIZE * _GRID_SIZE)
    grid_counts = [0] * (_GRID_SIZE * _GRID_SIZE)

    for y in range(frame.height):
        grid_y = min(_GRID_SIZE - 1, y * _GRID_SIZE // frame.height)
        for x in range(frame.width):
            pixel = y * frame.width + x
            offset = pixel * 3
            value = _luma(
                frame.rgb[offset],
                frame.rgb[offset + 1],
                frame.rgb[offset + 2],
            )
            lumas[pixel] = value
            grid_x = min(_GRID_SIZE - 1, x * _GRID_SIZE // frame.width)
            cell = grid_y * _GRID_SIZE + grid_x
            grid_sums[cell] += value
            grid_counts[cell] += 1

    # Very small frames can leave cells empty.  Empty cells receive zero and the
    # dimensions remain part of the descriptor so callers can reject incompatible
    # baselines rather than silently treating different sampling geometry as equal.
    grid = tuple(
        total / count if count else 0.0
        for total, count in zip(grid_sums, grid_counts, strict=True)
    )

    horizontal_total = 0.0
    horizontal_pairs = 0
    vertical_total = 0.0
    vertical_pairs = 0
    for y in range(frame.height):
        row = y * frame.width
        for x in range(frame.width):
            here = lumas[row + x]
            if x + 1 < frame.width:
                horizontal_total += abs(here - lumas[row + x + 1])
                horizontal_pairs += 1
            if y + 1 < frame.height:
                vertical_total += abs(here - lumas[row + frame.width + x])
                vertical_pairs += 1

    return SpatialFrameDescriptor(
        width=frame.width,
        height=frame.height,
        luma_grid=grid,
        horizontal_edge_energy=(
            horizontal_total / horizontal_pairs if horizontal_pairs else 0.0
        ),
        vertical_edge_energy=(
            vertical_total / vertical_pairs if vertical_pairs else 0.0
        ),
    )


def spatial_similarity(
    baseline: SpatialFrameDescriptor,
    candidate: SpatialFrameDescriptor,
) -> float:
    """Return normalized structural similarity for compatible sampling geometry."""

    if (baseline.width, baseline.height) != (candidate.width, candidate.height):
        return 0.0
    grid_delta = sum(
        abs(left - right)
        for left, right in zip(
            baseline.luma_grid, candidate.luma_grid, strict=True
        )
    ) / len(baseline.luma_grid)
    edge_delta = 0.5 * (
        abs(baseline.horizontal_edge_energy - candidate.horizontal_edge_energy)
        + abs(baseline.vertical_edge_energy - candidate.vertical_edge_energy)
    )
    # Grid placement carries more continuity information than total edge energy.
    delta = 0.8 * grid_delta + 0.2 * edge_delta
    return max(0.0, min(1.0, 1.0 - delta))


@dataclass(slots=True)
class SpatialContinuityMemory:
    """Last accepted measured spatial baseline per scene."""

    _accepted: dict[str, SpatialFrameDescriptor] = field(default_factory=dict)

    def latest(self, scene_id: str) -> SpatialFrameDescriptor | None:
        return self._accepted.get(scene_id)

    def accept(self, scene_id: str, descriptor: SpatialFrameDescriptor) -> None:
        if not scene_id:
            raise ValueError("spatial continuity memory requires a scene ID")
        self._accepted[scene_id] = descriptor

    def snapshot(self) -> dict[str, object]:
        return {
            "schema": SPATIAL_FRAME_EVIDENCE_SCHEMA,
            "accepted": {
                scene_id: descriptor.snapshot()
                for scene_id, descriptor in sorted(self._accepted.items())
            },
        }

    @classmethod
    def restore(cls, payload: dict[str, object]) -> SpatialContinuityMemory:
        if payload.get("schema") != SPATIAL_FRAME_EVIDENCE_SCHEMA:
            raise ValueError("unsupported spatial continuity memory schema")
        raw = payload.get("accepted", {})
        if not isinstance(raw, dict):
            raise ValueError("spatial continuity accepted state must be a mapping")
        memory = cls()
        for scene_id, descriptor_payload in raw.items():
            if not isinstance(scene_id, str) or not isinstance(descriptor_payload, dict):
                raise ValueError("invalid spatial continuity memory entry")
            memory.accept(scene_id, SpatialFrameDescriptor.restore(descriptor_payload))
        return memory


@dataclass(slots=True)
class MeasuredSpatialContinuityObserver:
    """Transactional spatial observer compatible with visual continuity QC.

    The measured score is mapped to ``environment`` because this observer detects
    frame-layout/environment continuity, not semantic identity.  It is intended to
    be combined with separately trained identity evidence before a production gate.
    """

    memory: SpatialContinuityMemory = field(default_factory=SpatialContinuityMemory)

    def observe(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> VisualContinuityObservation:
        if result.shot_id != plan.shot_id:
            raise ValueError("generated result and conditioning plan shot IDs must match")
        descriptor = describe_spatial_rgb_frame(result.image)
        baseline = self.memory.latest(plan.scene_id)
        score = 1.0 if baseline is None else spatial_similarity(baseline, descriptor)
        return VisualContinuityObservation(
            shot_id=plan.shot_id,
            scores={"environment": score},
            confidence=1.0,
        )

    def accept(
        self,
        result: NativeImageResearchResult,
        plan: NativeImageConditioningPlan,
    ) -> None:
        if result.shot_id != plan.shot_id:
            raise ValueError("generated result and conditioning plan shot IDs must match")
        self.memory.accept(plan.scene_id, describe_spatial_rgb_frame(result.image))
