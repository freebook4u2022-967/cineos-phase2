"""Measured spatial-detail integrity for assembled CINEOS native films.

Temporal QC catches black frames and frozen motion, but a failed decoder can also
produce non-black, time-varying frames with almost no spatial structure. This module
adds a separate fail-closed picture-integrity signal derived only from decoded movie
pixels. FFmpeg is used as a decoder/sampler, never as a visual generator.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SpatialIntegrityPolicy:
    """Versionable thresholds for measured spatial picture evidence."""

    min_variance: float = 12.0
    min_edge_mad: float = 1.0
    max_low_detail_ratio: float = 0.20

    def __post_init__(self) -> None:
        if self.min_variance < 0.0:
            raise ValueError("min_variance must be non-negative")
        if self.min_edge_mad < 0.0:
            raise ValueError("min_edge_mad must be non-negative")
        if not 0.0 <= self.max_low_detail_ratio <= 1.0:
            raise ValueError("max_low_detail_ratio must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SpatialIntegrityReport:
    """Auditable spatial-detail metrics from real decoded grayscale frames."""

    frame_count: int
    mean_variance: float
    mean_edge_mad: float
    low_detail_frame_ratio: float
    decision: str
    directives: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.decision in {"accept", "warn"}


def _variance(frame: bytes) -> float:
    if not frame:
        raise ValueError("sampled frame cannot be empty")
    mean = sum(frame) / len(frame)
    return sum((value - mean) ** 2 for value in frame) / len(frame)


def _edge_mad(frame: bytes, width: int, height: int) -> float:
    if width <= 1 or height <= 1:
        raise ValueError("spatial integrity requires dimensions greater than 1")
    if len(frame) != width * height:
        raise ValueError("sampled frame size does not match dimensions")

    total = 0.0
    comparisons = 0
    for row in range(height):
        base = row * width
        for column in range(width - 1):
            total += abs(frame[base + column] - frame[base + column + 1])
            comparisons += 1
    for row in range(height - 1):
        base = row * width
        next_base = (row + 1) * width
        for column in range(width):
            total += abs(frame[base + column] - frame[next_base + column])
            comparisons += 1
    return total / comparisons


def evaluate_spatial_samples(
    frames: Sequence[bytes],
    *,
    width: int,
    height: int,
    policy: SpatialIntegrityPolicy | None = None,
) -> SpatialIntegrityReport:
    """Reject featureless decoded output even when it is non-black and moving."""

    if not frames:
        raise ValueError("at least one sampled frame is required")
    expected = width * height
    if width <= 1 or height <= 1 or expected <= 0:
        raise ValueError("sample dimensions must be greater than 1")
    if any(len(frame) != expected for frame in frames):
        raise ValueError("all sampled frames must match the declared dimensions")

    active = policy or SpatialIntegrityPolicy()
    variances = tuple(_variance(frame) for frame in frames)
    edges = tuple(_edge_mad(frame, width, height) for frame in frames)
    low_detail = tuple(
        variance < active.min_variance and edge < active.min_edge_mad
        for variance, edge in zip(variances, edges, strict=True)
    )
    ratio = sum(low_detail) / len(low_detail)

    directives: list[str] = []
    if ratio > active.max_low_detail_ratio:
        decision = "reject"
        directives.append(
            "rerender low-detail picture regions with genuine spatial structure"
        )
    elif ratio > 0.0:
        decision = "warn"
        directives.append("review low-detail decoded frames for renderer collapse")
    else:
        decision = "accept"

    return SpatialIntegrityReport(
        frame_count=len(frames),
        mean_variance=sum(variances) / len(variances),
        mean_edge_mad=sum(edges) / len(edges),
        low_detail_frame_ratio=ratio,
        decision=decision,
        directives=tuple(directives),
    )


@dataclass(slots=True)
class FFmpegSpatialIntegrityEvaluator:
    """Sample real assembled-film pixels and measure spatial integrity."""

    sample_width: int = 32
    sample_height: int = 18
    sample_fps: float = 2.0
    ffmpeg_binary: str = "ffmpeg"
    policy: SpatialIntegrityPolicy = SpatialIntegrityPolicy()

    def __post_init__(self) -> None:
        if self.sample_width <= 1 or self.sample_height <= 1:
            raise ValueError("sample dimensions must be greater than 1")
        if self.sample_fps <= 0.0:
            raise ValueError("sample_fps must be positive")

    @property
    def frame_size(self) -> int:
        return self.sample_width * self.sample_height

    def evaluate(self, movie_path: str | Path) -> SpatialIntegrityReport:
        source = Path(movie_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        binary = shutil.which(self.ffmpeg_binary)
        if binary is None:
            raise RuntimeError(
                f"{self.ffmpeg_binary} is required for measured spatial film QC"
            )

        command = [
            binary,
            "-v",
            "error",
            "-i",
            str(source),
            "-vf",
            (
                f"fps={self.sample_fps},"
                f"scale={self.sample_width}:{self.sample_height}:flags=area,"
                "format=gray"
            ),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        completed = subprocess.run(command, check=True, capture_output=True)
        payload = completed.stdout
        if not payload:
            raise RuntimeError("ffmpeg produced no decoded spatial frame evidence")
        if len(payload) % self.frame_size:
            raise RuntimeError("ffmpeg returned an incomplete spatial sampled frame")

        frames = tuple(
            payload[offset : offset + self.frame_size]
            for offset in range(0, len(payload), self.frame_size)
        )
        return evaluate_spatial_samples(
            frames,
            width=self.sample_width,
            height=self.sample_height,
            policy=self.policy,
        )


__all__ = [
    "FFmpegSpatialIntegrityEvaluator",
    "SpatialIntegrityPolicy",
    "SpatialIntegrityReport",
    "evaluate_spatial_samples",
]
