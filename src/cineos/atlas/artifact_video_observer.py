"""Decode rendered video artifacts into production QC measurements.

The observer in this module measures the exact MP4 that Atlas produced. It
owns transport-level artifact and temporal checks and deliberately requires an
injected semantic scorer for character identity and motion quality. This keeps
low-level pixel heuristics from being mislabeled as semantic identity evidence.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .sequence_quality import CORE_METRICS, PRODUCTION_MEASUREMENT_SCHEMA


class VideoArtifactObservationError(RuntimeError):
    """Raised when a rendered artifact cannot produce trustworthy QC evidence."""


@dataclass(frozen=True, slots=True)
class RGBVideoSample:
    """Small decoded RGB sample taken from the actual rendered video artifact."""

    width: int
    height: int
    frames: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("video sample dimensions must be positive")
        expected = self.width * self.height * 3
        if not self.frames:
            raise ValueError("video sample requires at least one decoded frame")
        for frame in self.frames:
            if len(frame) != expected:
                raise ValueError(
                    f"decoded RGB frame expected {expected} bytes, got {len(frame)}"
                )


class VideoSampler(Protocol):
    def __call__(self, artifact: Path) -> RGBVideoSample: ...


class SemanticVideoScorer(Protocol):
    def __call__(
        self,
        sample: RGBVideoSample,
        *,
        artifact: Path,
        shot: Any,
        attempt_index: int,
    ) -> Mapping[str, float]: ...


@dataclass(frozen=True, slots=True)
class FFmpegRGBSampler:
    """Decode bounded RGB evidence through the system ffmpeg executable."""

    width: int = 96
    height: int = 96
    sample_fps: float = 4.0
    max_frames: int = 24
    ffmpeg_binary: str = "ffmpeg"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ffmpeg sample dimensions must be positive")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if self.max_frames < 2:
            raise ValueError("max_frames must be at least two")
        if not self.ffmpeg_binary.strip():
            raise ValueError("ffmpeg_binary must be non-empty")

    def __call__(self, artifact: Path) -> RGBVideoSample:
        if not artifact.is_file():
            raise VideoArtifactObservationError(
                f"rendered video artifact does not exist: {artifact}"
            )
        filter_graph = (
            f"fps={self.sample_fps},"
            f"scale={self.width}:{self.height}:flags=area"
        )
        command = [
            self.ffmpeg_binary,
            "-v",
            "error",
            "-i",
            str(artifact),
            "-an",
            "-sn",
            "-dn",
            "-vf",
            filter_graph,
            "-frames:v",
            str(self.max_frames),
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise VideoArtifactObservationError(
                f"ffmpeg executable not found: {self.ffmpeg_binary}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise VideoArtifactObservationError(
                f"ffmpeg failed to decode rendered artifact: {detail or exc.returncode}"
            ) from exc

        frame_bytes = self.width * self.height * 3
        payload = completed.stdout
        if not payload or len(payload) % frame_bytes:
            raise VideoArtifactObservationError(
                "ffmpeg returned empty or incomplete RGB frame evidence"
            )
        frames = tuple(
            payload[offset : offset + frame_bytes]
            for offset in range(0, len(payload), frame_bytes)
        )
        if len(frames) < 2:
            raise VideoArtifactObservationError(
                "production temporal QC requires at least two decoded video frames"
            )
        return RGBVideoSample(self.width, self.height, frames)


def _artifact_integrity(sample: RGBVideoSample) -> float:
    """Detect catastrophic black/clipped sampled frames without aesthetic claims."""

    pixels_per_frame = sample.width * sample.height
    corrupted = 0
    for frame in sample.frames:
        black = clipped = 0
        for offset in range(0, len(frame), 3):
            red, green, blue = frame[offset : offset + 3]
            if max(red, green, blue) <= 4:
                black += 1
            if max(red, green, blue) >= 251:
                clipped += 1
        if black / pixels_per_frame >= 0.985 or clipped / pixels_per_frame >= 0.985:
            corrupted += 1
    return max(0.0, 1.0 - corrupted / len(sample.frames))


def _temporal_consistency(sample: RGBVideoSample) -> float:
    """Penalize catastrophic adjacent-frame discontinuities in decoded RGB evidence."""

    if len(sample.frames) < 2:
        return 0.0
    discontinuities: list[float] = []
    for previous, current in zip(sample.frames, sample.frames[1:]):
        mean_delta = sum(
            abs(left - right) for left, right in zip(previous, current, strict=True)
        ) / (len(current) * 255.0)
        # Normal movement is deliberately not punished. Only very large average
        # RGB jumps are treated as transport/temporal instability here; semantic
        # temporal reasoning belongs in stronger model observers.
        discontinuity = max(0.0, min(1.0, (mean_delta - 0.45) / 0.55))
        discontinuities.append(discontinuity)
    return max(0.0, 1.0 - sum(discontinuities) / len(discontinuities))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactVideoMetricObserver:
    """Create artifact-bound production measurements from decoded video evidence.

    ``semantic_scorer`` receives the decoded frames and must return at least
    ``identity_similarity`` and ``motion_quality``. Optional semantic metrics
    such as anatomy, object interaction, and lip-sync are preserved. The
    observer itself supplies conservative artifact-integrity and RGB temporal
    evidence, then binds the complete report to the exact rendered artifact.
    """

    def __init__(
        self,
        semantic_scorer: SemanticVideoScorer,
        *,
        sampler: VideoSampler | None = None,
        observer_id: str = "cineos-artifact-video-observer/0.1",
    ) -> None:
        if not callable(semantic_scorer):
            raise TypeError("semantic_scorer must be callable")
        if sampler is not None and not callable(sampler):
            raise TypeError("sampler must be callable")
        if not observer_id.strip():
            raise ValueError("observer_id must be non-empty")
        self.semantic_scorer = semantic_scorer
        self.sampler = sampler or FFmpegRGBSampler()
        self.observer_id = observer_id.strip()

    def __call__(
        self,
        output_path: str,
        *,
        shot: Any,
        attempt_index: int,
    ) -> dict[str, Any]:
        artifact = Path(output_path)
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise VideoArtifactObservationError(
                f"rendered video artifact is missing or empty: {artifact}"
            )
        sample = self.sampler(artifact)
        semantic = self.semantic_scorer(
            sample,
            artifact=artifact,
            shot=shot,
            attempt_index=attempt_index,
        )
        if not isinstance(semantic, Mapping):
            raise VideoArtifactObservationError(
                "semantic video scorer must return a metric mapping"
            )
        metrics: dict[str, float] = {
            "artifact_integrity": _artifact_integrity(sample),
            "temporal_consistency": _temporal_consistency(sample),
        }
        for name, value in semantic.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if not 0.0 <= numeric <= 1.0:
                raise VideoArtifactObservationError(
                    f"semantic metric {name!r} must be between 0 and 1"
                )
            metrics[name] = numeric

        semantic_required = ("identity_similarity", "motion_quality")
        missing_semantic = [name for name in semantic_required if name not in metrics]
        if missing_semantic:
            raise VideoArtifactObservationError(
                "semantic video scorer missing required metric(s): "
                + ", ".join(missing_semantic)
            )
        missing_core = [name for name in CORE_METRICS if name not in metrics]
        if missing_core:
            raise VideoArtifactObservationError(
                "video observer missing core metric(s): " + ", ".join(missing_core)
            )
        return {
            "schema": PRODUCTION_MEASUREMENT_SCHEMA,
            "observer_id": self.observer_id,
            "artifact_sha256": _sha256_file(artifact),
            "metrics": metrics,
            "sample": {
                "width": sample.width,
                "height": sample.height,
                "frame_count": len(sample.frames),
            },
        }


__all__ = [
    "ArtifactVideoMetricObserver",
    "FFmpegRGBSampler",
    "RGBVideoSample",
    "SemanticVideoScorer",
    "VideoArtifactObservationError",
    "VideoSampler",
]
