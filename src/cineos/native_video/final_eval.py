"""Measured final-film temporal evaluation for CINEOS native video.

This module keeps quality claims tied to decoded frame evidence. The pure metric
layer accepts sampled grayscale frames and is dependency-free for CI; the FFmpeg
sampler is an explicit production adapter that extracts real pixels from a movie.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TemporalFilmEvalPolicy:
    """Versionable thresholds for final-film temporal evidence."""

    black_luma: float = 8.0
    frozen_mad: float = 0.75
    hard_cut_mad: float = 42.0
    max_black_ratio: float = 0.02
    max_frozen_ratio: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 <= self.black_luma <= 255.0:
            raise ValueError("black_luma must be in [0, 255]")
        if min(self.frozen_mad, self.hard_cut_mad) < 0.0:
            raise ValueError("MAD thresholds must be non-negative")
        if self.frozen_mad > self.hard_cut_mad:
            raise ValueError("frozen_mad cannot exceed hard_cut_mad")
        if not 0.0 <= self.max_black_ratio <= 1.0:
            raise ValueError("max_black_ratio must be in [0, 1]")
        if not 0.0 <= self.max_frozen_ratio <= 1.0:
            raise ValueError("max_frozen_ratio must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TemporalFilmEvalReport:
    """Auditable temporal metrics derived from sampled decoded movie frames."""

    frame_count: int
    mean_luma: float
    mean_variance: float
    mean_interframe_mad: float
    black_frame_ratio: float
    frozen_transition_ratio: float
    hard_cut_transition_ratio: float
    decision: str
    directives: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return self.decision in {"accept", "warn"}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(frame: bytes) -> float:
    if not frame:
        raise ValueError("sampled frame cannot be empty")
    average = sum(frame) / len(frame)
    return sum((value - average) ** 2 for value in frame) / len(frame)


def _mad(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("sampled frames must have identical non-zero sizes")
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) / len(left)


def evaluate_sampled_frames(
    frames: Sequence[bytes],
    policy: TemporalFilmEvalPolicy | None = None,
) -> TemporalFilmEvalReport:
    """Evaluate decoded grayscale frames without trusting container metadata."""

    if not frames:
        raise ValueError("at least one sampled frame is required")
    frame_size = len(frames[0])
    if frame_size <= 0 or any(len(frame) != frame_size for frame in frames):
        raise ValueError("all sampled frames must have the same non-zero size")

    active_policy = policy or TemporalFilmEvalPolicy()
    lumas = [sum(frame) / len(frame) for frame in frames]
    variances = [_variance(frame) for frame in frames]
    transitions = [_mad(left, right) for left, right in zip(frames, frames[1:])]

    black_ratio = sum(luma <= active_policy.black_luma for luma in lumas) / len(frames)
    transition_count = max(1, len(transitions))
    frozen_ratio = (
        sum(value <= active_policy.frozen_mad for value in transitions)
        / transition_count
        if transitions
        else 0.0
    )
    hard_cut_ratio = (
        sum(value >= active_policy.hard_cut_mad for value in transitions)
        / transition_count
        if transitions
        else 0.0
    )

    directives: list[str] = []
    decision = "accept"
    if black_ratio > active_policy.max_black_ratio:
        decision = "reject"
        directives.append("rerender or replace black/near-black frame regions")
    if frozen_ratio > active_policy.max_frozen_ratio:
        decision = "reject"
        directives.append("rerender frozen temporal regions with real motion evidence")
    if decision == "accept" and hard_cut_ratio > 0.5:
        decision = "warn"
        directives.append(
            "review excessive hard-cut evidence against the intended edit plan"
        )

    return TemporalFilmEvalReport(
        frame_count=len(frames),
        mean_luma=_mean(lumas),
        mean_variance=_mean(variances),
        mean_interframe_mad=_mean(transitions),
        black_frame_ratio=black_ratio,
        frozen_transition_ratio=frozen_ratio,
        hard_cut_transition_ratio=hard_cut_ratio,
        decision=decision,
        directives=tuple(directives),
    )


@dataclass(slots=True)
class FFmpegTemporalFilmEvaluator:
    """Extract real decoded grayscale pixels and run final-film temporal QC."""

    sample_width: int = 32
    sample_height: int = 18
    sample_fps: float = 2.0
    ffmpeg_binary: str = "ffmpeg"
    policy: TemporalFilmEvalPolicy = TemporalFilmEvalPolicy()

    def __post_init__(self) -> None:
        if self.sample_width <= 0 or self.sample_height <= 0:
            raise ValueError("sample dimensions must be positive")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")

    @property
    def frame_size(self) -> int:
        return self.sample_width * self.sample_height

    def evaluate(self, movie_path: str | Path) -> TemporalFilmEvalReport:
        source = Path(movie_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        binary = shutil.which(self.ffmpeg_binary)
        if binary is None:
            raise RuntimeError(
                f"{self.ffmpeg_binary} is required for measured temporal film QC"
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
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        payload = completed.stdout
        if not payload:
            raise RuntimeError("ffmpeg produced no decoded frame evidence")
        if len(payload) % self.frame_size:
            raise RuntimeError("ffmpeg returned an incomplete sampled frame")

        frames = tuple(
            payload[offset : offset + self.frame_size]
            for offset in range(0, len(payload), self.frame_size)
        )
        return evaluate_sampled_frames(frames, self.policy)
