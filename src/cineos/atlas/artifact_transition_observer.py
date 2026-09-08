"""Measure cross-shot visual continuity from the exact rendered artifacts.

The observer decodes the terminal window of the predecessor and the initial window
of the successor, then compares pinned external SigLIP2 image features across the
boundary. CINEOS owns the sampling, artifact binding and acceptance policy; the
learned visual representation remains explicitly external pretrained capability.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .artifact_video_observer import RGBVideoSample
from .transition_quality import TRANSITION_QUALITY_SCHEMA


class TransitionArtifactObservationError(RuntimeError):
    """Raised when exact boundary artifacts cannot yield measured continuity evidence."""


class BoundaryFeatureScorer(Protocol):
    semantic_measurement_evidence: bool

    def encode_sample_features(
        self, sample: RGBVideoSample
    ) -> tuple[tuple[float, ...], ...]: ...


class BoundarySampler(Protocol):
    production_measurement_evidence: bool

    def __call__(self, artifact: Path, *, tail: bool) -> RGBVideoSample: ...


@dataclass(frozen=True, slots=True)
class FFmpegBoundaryRGBSampler:
    """Decode a bounded one-second tail/head window from an actual video artifact."""

    width: int = 128
    height: int = 128
    sample_fps: float = 4.0
    window_seconds: float = 1.0
    max_frames: int = 4
    ffmpeg_binary: str = "ffmpeg"

    production_measurement_evidence = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("boundary sample dimensions must be positive")
        if self.sample_fps <= 0 or self.window_seconds <= 0:
            raise ValueError("boundary sample timing must be positive")
        if self.max_frames < 2:
            raise ValueError("boundary sample requires at least two frames")
        if not self.ffmpeg_binary.strip():
            raise ValueError("ffmpeg_binary must be non-empty")

    def __call__(self, artifact: Path, *, tail: bool) -> RGBVideoSample:
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise TransitionArtifactObservationError(
                f"transition artifact is missing or empty: {artifact}"
            )
        command = [self.ffmpeg_binary, "-v", "error"]
        if tail:
            command.extend(["-sseof", f"-{self.window_seconds:.6f}"])
        command.extend(["-i", str(artifact)])
        if not tail:
            command.extend(["-t", f"{self.window_seconds:.6f}"])
        command.extend(
            [
                "-an",
                "-sn",
                "-dn",
                "-vf",
                f"fps={self.sample_fps},scale={self.width}:{self.height}:flags=area",
                "-frames:v",
                str(self.max_frames),
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
        )
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=30,
            )
        except FileNotFoundError as exc:
            raise TransitionArtifactObservationError(
                f"ffmpeg executable not found: {self.ffmpeg_binary}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise TransitionArtifactObservationError(
                "ffmpeg transition sampling exceeded 30 seconds"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise TransitionArtifactObservationError(
                f"ffmpeg failed to decode transition artifact: {detail or exc.returncode}"
            ) from exc

        frame_bytes = self.width * self.height * 3
        payload = completed.stdout
        if not payload or len(payload) % frame_bytes:
            raise TransitionArtifactObservationError(
                "ffmpeg returned empty or incomplete transition frame evidence"
            )
        frames = tuple(
            payload[offset : offset + frame_bytes]
            for offset in range(0, len(payload), frame_bytes)
        )
        if len(frames) < 2:
            raise TransitionArtifactObservationError(
                "transition measurement requires at least two frames on each side"
            )
        return RGBVideoSample(self.width, self.height, frames)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right) or not left:
        raise TransitionArtifactObservationError(
            "transition feature vectors must be non-empty and dimension matched"
        )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        raise TransitionArtifactObservationError("transition feature vector has zero norm")
    value = sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )
    if not math.isfinite(value):
        raise TransitionArtifactObservationError("transition feature cosine is non-finite")
    return max(-1.0, min(1.0, value))


def _feature_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return (_cosine(left, right) + 1.0) / 2.0


def _feature_step(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return max(0.0, min(1.0, 1.0 - _cosine(left, right)))


def _boundary_metrics(
    previous_features: tuple[tuple[float, ...], ...],
    current_features: tuple[tuple[float, ...], ...],
) -> tuple[float, float]:
    """Return learned seam similarity and local motion-step consistency.

    Motion consistency compares the cross-boundary feature displacement with the
    immediately adjacent within-shot displacements. This detects cut-like semantic
    jumps without pretending that low-level pixel deltas constitute identity evidence.
    """

    if len(previous_features) < 2 or len(current_features) < 2:
        raise TransitionArtifactObservationError(
            "transition feature measurement requires two frames on each side"
        )
    previous_step = _feature_step(previous_features[-2], previous_features[-1])
    boundary_step = _feature_step(previous_features[-1], current_features[0])
    current_step = _feature_step(current_features[0], current_features[1])
    local_step = (previous_step + current_step) / 2.0
    # A 0.5 feature-space step mismatch is treated as maximally inconsistent. The
    # transition policy remains the versioned authority over whether this normalized
    # metric is acceptable for production.
    motion_consistency = 1.0 - min(1.0, abs(boundary_step - local_step) / 0.5)
    return (
        _feature_similarity(previous_features[-1], current_features[0]),
        max(0.0, min(1.0, motion_consistency)),
    )


class SigLIP2ArtifactTransitionObserver:
    """Artifact-bound cross-shot observer using the pinned production SigLIP2 encoder."""

    observer_id = "cineos-siglip2-artifact-transition-observer/0.1"

    def __init__(
        self,
        feature_scorer: BoundaryFeatureScorer,
        *,
        sampler: BoundarySampler | None = None,
    ) -> None:
        if not callable(getattr(feature_scorer, "encode_sample_features", None)):
            raise TypeError("transition feature scorer must expose encode_sample_features")
        if sampler is not None and not callable(sampler):
            raise TypeError("transition sampler must be callable")
        self.feature_scorer = feature_scorer
        self.sampler = sampler or FFmpegBoundaryRGBSampler()
        # Injected samplers are useful for unit tests/research but cannot promote
        # synthetic frame evidence into production continuity attestation.
        self.production_measurement_evidence = (
            getattr(feature_scorer, "semantic_measurement_evidence", False) is True
            and sampler is None
            and getattr(self.sampler, "production_measurement_evidence", False) is True
        )

    def __call__(
        self,
        previous_path: str,
        current_path: str,
        *,
        previous_shot: Any,
        current_shot: Any,
        attempt_index: int,
    ) -> dict[str, Any]:
        del previous_shot, current_shot, attempt_index
        previous = Path(previous_path)
        current = Path(current_path)
        previous_sample = self.sampler(previous, tail=True)
        current_sample = self.sampler(current, tail=False)
        previous_features = self.feature_scorer.encode_sample_features(previous_sample)
        current_features = self.feature_scorer.encode_sample_features(current_sample)
        visual, motion = _boundary_metrics(previous_features, current_features)
        return {
            "schema": TRANSITION_QUALITY_SCHEMA,
            "production_measurement_evidence": self.production_measurement_evidence,
            "observer_id": self.observer_id,
            "previous_output_sha256": _sha256_file(previous),
            "current_output_sha256": _sha256_file(current),
            "measured_sample_count": min(
                len(previous_features), len(current_features)
            ),
            "metrics": {
                "visual_seam_similarity": visual,
                "motion_boundary_consistency": motion,
            },
        }


__all__ = [
    "FFmpegBoundaryRGBSampler",
    "SigLIP2ArtifactTransitionObserver",
    "TransitionArtifactObservationError",
]
