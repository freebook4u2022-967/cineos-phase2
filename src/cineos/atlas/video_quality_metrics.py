"""Measured quality metrics extracted from real rendered video artifacts.

The connected-shot quality gate must inspect generated pixels rather than accept
synthetic scores. This module provides a foundation-neutral extractor for
artifact integrity, within-shot temporal consistency, and motion coherence from
sampled decoded RGB frames. Semantic character identity is intentionally supplied
by an injected identity observer; missing identity evidence fails closed rather
than being inferred from low-level pixels.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from cineos.native_image.neural_decoder import DecodedRGBFrame
from cineos.native_image.pixel_observer import PixelFrameDescriptor, describe_rgb_frame


class VideoQualityMetricError(RuntimeError):
    """Raised when a rendered artifact cannot produce trustworthy metrics."""


class IdentityMetricSource(Protocol):
    """Semantic identity observer used alongside low-level video measurements."""

    def __call__(
        self,
        output_path: str,
        *,
        shot: Any,
        frames: tuple[DecodedRGBFrame, ...],
        attempt_index: int,
    ) -> float: ...


FrameReader = Callable[[str], Iterable[Any]]


def _default_frame_reader(output_path: str) -> Iterable[Any]:
    try:
        import imageio.v3 as iio
    except ImportError as exc:  # pragma: no cover - exercised in video environment
        raise VideoQualityMetricError(
            "video metric extraction requires the cineos[video] dependencies"
        ) from exc
    try:
        return iio.imiter(output_path, plugin="ffmpeg")
    except Exception as exc:  # pragma: no cover - backend-specific failure
        raise VideoQualityMetricError(
            f"cannot open rendered video: {output_path}"
        ) from exc


def _coerce_rgb_frame(frame: Any) -> DecodedRGBFrame:
    shape = getattr(frame, "shape", None)
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise VideoQualityMetricError("decoded video frame must have HxWxC shape")
    height, width, channels = (int(value) for value in shape)
    if width <= 0 or height <= 0 or channels < 3:
        raise VideoQualityMetricError("decoded video frame has invalid dimensions")
    try:
        rgb_source = frame if channels == 3 else frame[..., :3]
        payload = rgb_source.tobytes()
    except Exception as exc:
        raise VideoQualityMetricError(
            "decoded video frame cannot expose RGB bytes"
        ) from exc
    expected = width * height * 3
    if len(payload) != expected:
        raise VideoQualityMetricError(
            f"decoded RGB payload expected {expected} bytes, got {len(payload)}"
        )
    return DecodedRGBFrame(width=width, height=height, rgb=payload)


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _histogram_similarity(
    left: PixelFrameDescriptor, right: PixelFrameDescriptor
) -> float:
    distance = 0.5 * sum(
        abs(a - b)
        for a, b in zip(left.luma_histogram, right.luma_histogram, strict=True)
    )
    return _bounded(1.0 - distance)


def _color_similarity(left: PixelFrameDescriptor, right: PixelFrameDescriptor) -> float:
    distance = (
        abs(left.mean_red - right.mean_red)
        + abs(left.mean_green - right.mean_green)
        + abs(left.mean_blue - right.mean_blue)
    ) / 3.0
    return _bounded(1.0 - distance)


def _spatial_similarity(
    left: PixelFrameDescriptor, right: PixelFrameDescriptor
) -> float:
    distance = sum(
        abs(a - b) for a, b in zip(left.spatial_luma, right.spatial_luma, strict=True)
    ) / len(left.spatial_luma)
    return _bounded(1.0 - distance)


def _pair_temporal_similarity(
    left: PixelFrameDescriptor, right: PixelFrameDescriptor
) -> float:
    lighting_delta = abs(left.mean_luma - right.mean_luma) + 2.0 * abs(
        left.luma_std - right.luma_std
    )
    lighting = _bounded(1.0 - lighting_delta)
    return (
        0.35 * _histogram_similarity(left, right)
        + 0.25 * _color_similarity(left, right)
        + 0.25 * _spatial_similarity(left, right)
        + 0.15 * lighting
    )


def _motion_magnitude(left: PixelFrameDescriptor, right: PixelFrameDescriptor) -> float:
    spatial = sum(
        abs(a - b) for a, b in zip(left.spatial_luma, right.spatial_luma, strict=True)
    ) / len(left.spatial_luma)
    edge = abs(left.edge_energy - right.edge_energy)
    histogram = 1.0 - _histogram_similarity(left, right)
    return _bounded(0.55 * spatial + 0.25 * edge + 0.20 * histogram)


def _artifact_integrity(descriptors: list[PixelFrameDescriptor]) -> float:
    penalties: list[float] = []
    for descriptor in descriptors:
        black_penalty = _bounded((descriptor.black_fraction - 0.92) / 0.08)
        clip_penalty = _bounded((descriptor.clipped_fraction - 0.50) / 0.50)
        flat_penalty = _bounded((0.004 - descriptor.luma_std) / 0.004)
        penalties.append(max(black_penalty, clip_penalty, 0.35 * flat_penalty))
    return _bounded(1.0 - sum(penalties) / len(penalties))


def _temporal_consistency(descriptors: list[PixelFrameDescriptor]) -> float:
    if len(descriptors) < 2:
        return 0.0
    similarities = [
        _pair_temporal_similarity(left, right)
        for left, right in zip(descriptors, descriptors[1:], strict=False)
    ]
    return sum(similarities) / len(similarities)


def _motion_quality(descriptors: list[PixelFrameDescriptor]) -> float:
    if len(descriptors) < 2:
        return 0.0
    magnitudes = [
        _motion_magnitude(left, right)
        for left, right in zip(descriptors, descriptors[1:], strict=False)
    ]
    if len(magnitudes) == 1:
        return 1.0
    jerks = [
        abs(left - right)
        for left, right in zip(magnitudes, magnitudes[1:], strict=False)
    ]
    mean_jerk = sum(jerks) / len(jerks)
    catastrophic_jump = max(magnitudes)
    jump_penalty = _bounded((catastrophic_jump - 0.65) / 0.35)
    return _bounded(1.0 - 3.5 * mean_jerk - 0.5 * jump_penalty)


@dataclass(slots=True)
class MeasuredVideoQualityExtractor:
    """Extract auditable quality metrics from a rendered MP4-like artifact.

    Frames are sampled incrementally to bound CPU and memory use. Pixel-derived
    metrics deliberately avoid making semantic identity claims. ``identity_source``
    must inspect the same sampled frames (or richer external evidence) and return
    a normalized semantic identity score.
    """

    identity_source: IdentityMetricSource
    frame_reader: FrameReader = _default_frame_reader
    sample_stride: int = 4
    max_samples: int = 16
    min_samples: int = 3

    def __post_init__(self) -> None:
        if not callable(self.identity_source):
            raise TypeError("identity_source must be callable")
        if not callable(self.frame_reader):
            raise TypeError("frame_reader must be callable")
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be positive")
        if self.max_samples < 2:
            raise ValueError("max_samples must be at least 2")
        if not 2 <= self.min_samples <= self.max_samples:
            raise ValueError("min_samples must be between 2 and max_samples")

    def __call__(
        self,
        output_path: str,
        *,
        shot: Any,
        attempt_index: int,
    ) -> dict[str, float]:
        path = Path(output_path)
        if not path.exists() or not path.is_file():
            raise VideoQualityMetricError(f"rendered artifact does not exist: {path}")
        if path.stat().st_size <= 0:
            raise VideoQualityMetricError(f"rendered artifact is empty: {path}")

        sampled: list[DecodedRGBFrame] = []
        try:
            frames = self.frame_reader(str(path))
            for index, frame in enumerate(frames):
                if index % self.sample_stride:
                    continue
                sampled.append(_coerce_rgb_frame(frame))
                if len(sampled) >= self.max_samples:
                    break
        except VideoQualityMetricError:
            raise
        except Exception as exc:
            raise VideoQualityMetricError(
                f"cannot decode rendered artifact: {path}"
            ) from exc

        if len(sampled) < self.min_samples:
            raise VideoQualityMetricError(
                f"rendered artifact yielded {len(sampled)} sampled frames; "
                f"at least {self.min_samples} are required"
            )
        dimensions = {(frame.width, frame.height) for frame in sampled}
        if len(dimensions) != 1:
            raise VideoQualityMetricError(
                "sampled video frames changed dimensions within one shot"
            )

        descriptors = [describe_rgb_frame(frame) for frame in sampled]
        identity = self.identity_source(
            str(path),
            shot=shot,
            frames=tuple(sampled),
            attempt_index=attempt_index,
        )
        if isinstance(identity, bool) or not isinstance(identity, (int, float)):
            raise VideoQualityMetricError(
                "identity_source must return a numeric score in [0, 1]"
            )
        identity_score = float(identity)
        if not 0.0 <= identity_score <= 1.0:
            raise VideoQualityMetricError(
                "identity_source must return a score in [0, 1]"
            )

        return {
            "identity_similarity": identity_score,
            "temporal_consistency": _temporal_consistency(descriptors),
            "artifact_integrity": _artifact_integrity(descriptors),
            "motion_quality": _motion_quality(descriptors),
        }


__all__ = [
    "IdentityMetricSource",
    "MeasuredVideoQualityExtractor",
    "VideoQualityMetricError",
]
