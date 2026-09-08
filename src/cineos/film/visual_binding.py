"""Measured binding between approved connected shots and final-film video.

Delivery encoding may transcode the approved shot artifacts, so byte equality cannot
prove visual identity. This module independently decodes the ordered approved shots and
final film to the same low-rate RGB representation and requires strong pixel
correlation with a tightly bounded frame-alignment search. RGB is intentionally used
instead of luma-only sampling so a re-signed delivery cannot silently substitute or
radically alter chroma while preserving grayscale structure. Correlation is paired with
a bounded mean absolute pixel error so affine brightness/contrast substitutions cannot
pass merely because they preserve correlation. The sampled duration must also match
within the bounded alignment tolerance so approved footage cannot be padded with or
truncated around unapproved video. It is an integrity gate, not an aesthetic-quality
metric.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VISUAL_BINDING_SCHEMA = "cineos-production-visual-binding/0.5"
VISUAL_BINDING_SAMPLE_FPS = 8
VISUAL_BINDING_WIDTH = 32
VISUAL_BINDING_HEIGHT = 18
VISUAL_BINDING_CHANNELS = 3
VISUAL_BINDING_FRAME_BYTES = (
    VISUAL_BINDING_WIDTH * VISUAL_BINDING_HEIGHT * VISUAL_BINDING_CHANNELS
)
VISUAL_BINDING_MAX_LAG_FRAMES = 1
VISUAL_BINDING_MAX_FRAME_DELTA = VISUAL_BINDING_MAX_LAG_FRAMES
MIN_VISUAL_BINDING_CORRELATION = 0.92
MAX_VISUAL_BINDING_MEAN_ABSOLUTE_ERROR = 24.0
MIN_VISUAL_BINDING_FRAMES = 4


class VisualBindingError(RuntimeError):
    """Raised when final-film video cannot be bound to approved connected shots."""


@dataclass(frozen=True, slots=True)
class VisualBindingEvidence:
    """Measured evidence that final-film video derives from approved ordered shots."""

    source_sha256: tuple[str, ...]
    final_artifact_sha256: str
    sample_fps: int
    width: int
    height: int
    channels: int
    approved_sampled_frames: int
    final_sampled_frames: int
    max_frame_delta: int
    compared_frames: int
    alignment_lag_frames: int
    correlation: float
    minimum_correlation: float
    mean_absolute_error: float
    maximum_mean_absolute_error: float
    evidence_sha256: str

    @property
    def accepted(self) -> bool:
        return (
            self.correlation >= self.minimum_correlation
            and self.mean_absolute_error <= self.maximum_mean_absolute_error
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": VISUAL_BINDING_SCHEMA,
            "source_sha256": list(self.source_sha256),
            "final_artifact_sha256": self.final_artifact_sha256,
            "sample_fps": self.sample_fps,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "approved_sampled_frames": self.approved_sampled_frames,
            "final_sampled_frames": self.final_sampled_frames,
            "max_frame_delta": self.max_frame_delta,
            "compared_frames": self.compared_frames,
            "alignment_lag_frames": self.alignment_lag_frames,
            "correlation": self.correlation,
            "minimum_correlation": self.minimum_correlation,
            "mean_absolute_error": self.mean_absolute_error,
            "maximum_mean_absolute_error": self.maximum_mean_absolute_error,
            "accepted": self.accepted,
            "evidence_sha256": self.evidence_sha256,
        }


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise VisualBindingError(
            "FFmpeg is unavailable; install ffmpeg for production visual binding"
        )
    return executable


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VisualBindingError(
            f"cannot hash visual-binding artifact: {path}"
        ) from exc
    return digest.hexdigest()


def _decode_binding_rgb(
    path: Path,
    *,
    duration_seconds: float | None = None,
) -> bytes:
    source = path.resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise VisualBindingError(f"missing or empty visual-binding artifact: {source}")
    command = [
        _ffmpeg(),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
    ]
    if duration_seconds is not None:
        try:
            duration = float(duration_seconds)
        except (TypeError, ValueError) as exc:
            raise VisualBindingError(
                "visual-binding edit duration must be finite and positive"
            ) from exc
        if not math.isfinite(duration) or duration <= 0:
            raise VisualBindingError(
                "visual-binding edit duration must be finite and positive"
            )
        command.extend(["-t", f"{duration:.9f}"])
    command.extend(
        [
            "-vf",
            (
                f"fps={VISUAL_BINDING_SAMPLE_FPS},"
                f"scale={VISUAL_BINDING_WIDTH}:{VISUAL_BINDING_HEIGHT}:"
                "flags=area,format=rgb24"
            ),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    )
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise VisualBindingError(f"FFmpeg visual-binding decode failed: {stderr}")
    payload = bytes(result.stdout)
    if not payload or len(payload) % VISUAL_BINDING_FRAME_BYTES:
        raise VisualBindingError("FFmpeg returned malformed visual-binding frames")
    return payload


def _pixel_correlation(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        raise VisualBindingError(
            "visual-binding correlation requires equal non-empty pixel spans"
        )
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = 0.0
    left_energy = 0.0
    right_energy = 0.0
    for left_value, right_value in zip(left, right):
        left_delta = left_value - left_mean
        right_delta = right_value - right_mean
        numerator += left_delta * right_delta
        left_energy += left_delta * left_delta
        right_energy += right_delta * right_delta
    denominator = math.sqrt(left_energy * right_energy)
    if denominator <= 0 or not math.isfinite(denominator):
        raise VisualBindingError(
            "visual-binding frames have insufficient non-constant image signal"
        )
    value = numerator / denominator
    if not math.isfinite(value):
        raise VisualBindingError("visual-binding correlation is non-finite")
    return max(-1.0, min(1.0, value))


def _mean_absolute_error(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        raise VisualBindingError(
            "visual-binding error metric requires equal non-empty pixel spans"
        )
    value = sum(
        abs(left_value - right_value) for left_value, right_value in zip(left, right)
    ) / len(left)
    if not math.isfinite(value):
        raise VisualBindingError("visual-binding mean absolute error is non-finite")
    return value


def _best_alignment(approved: bytes, encoded: bytes) -> tuple[float, int, int, float]:
    frame_bytes = VISUAL_BINDING_FRAME_BYTES
    approved_frames = len(approved) // frame_bytes
    encoded_frames = len(encoded) // frame_bytes
    frame_delta = abs(approved_frames - encoded_frames)
    if frame_delta > VISUAL_BINDING_MAX_FRAME_DELTA:
        raise VisualBindingError(
            "final-film sampled duration does not match approved connected shots: "
            f"approved_frames={approved_frames}, final_frames={encoded_frames}, "
            f"allowed_delta<={VISUAL_BINDING_MAX_FRAME_DELTA}"
        )
    best: tuple[float, int, int, float] | None = None
    for lag in range(-VISUAL_BINDING_MAX_LAG_FRAMES, VISUAL_BINDING_MAX_LAG_FRAMES + 1):
        if lag >= 0:
            left_start = lag * frame_bytes
            right_start = 0
            available_left = approved_frames - lag
            available_right = encoded_frames
        else:
            left_start = 0
            right_start = (-lag) * frame_bytes
            available_left = approved_frames
            available_right = encoded_frames + lag
        compared_frames = min(available_left, available_right)
        if compared_frames < MIN_VISUAL_BINDING_FRAMES:
            continue
        span = compared_frames * frame_bytes
        left_span = approved[left_start : left_start + span]
        right_span = encoded[right_start : right_start + span]
        score = _pixel_correlation(left_span, right_span)
        mean_absolute_error = _mean_absolute_error(left_span, right_span)
        candidate = (score, lag, compared_frames, mean_absolute_error)
        if best is None or score > best[0]:
            best = candidate
    if best is None:
        raise VisualBindingError(
            "approved shots and final film have no sufficient overlapping visual span"
        )
    return best


def _evidence_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def measure_visual_binding(
    approved_shots: Sequence[str | Path],
    final_artifact: str | Path,
    *,
    durations_seconds: Sequence[float] | None = None,
    minimum_correlation: float = MIN_VISUAL_BINDING_CORRELATION,
    maximum_mean_absolute_error: float = MAX_VISUAL_BINDING_MEAN_ABSOLUTE_ERROR,
) -> VisualBindingEvidence:
    """Measure whether final-film decoded RGB video matches approved shots in order."""
    if not approved_shots:
        raise VisualBindingError("visual binding requires at least one approved shot")
    if durations_seconds is not None and len(durations_seconds) != len(approved_shots):
        raise VisualBindingError(
            "visual-binding edit duration count does not match approved shot count"
        )
    try:
        threshold = float(minimum_correlation)
    except (TypeError, ValueError) as exc:
        raise VisualBindingError("visual-binding threshold must be finite") from exc
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise VisualBindingError(
            "visual-binding threshold must be in the interval (0, 1]"
        )
    try:
        max_error = float(maximum_mean_absolute_error)
    except (TypeError, ValueError) as exc:
        raise VisualBindingError(
            "visual-binding maximum mean absolute error must be finite"
        ) from exc
    if not math.isfinite(max_error) or not 0.0 <= max_error <= 255.0:
        raise VisualBindingError(
            "visual-binding maximum mean absolute error must be in the interval [0, 255]"
        )

    source_paths = tuple(Path(path).resolve() for path in approved_shots)
    decoded_parts = []
    for index, path in enumerate(source_paths):
        duration = None if durations_seconds is None else durations_seconds[index]
        decoded_parts.append(_decode_binding_rgb(path, duration_seconds=duration))
    approved_video = b"".join(decoded_parts)
    final_path = Path(final_artifact).resolve()
    final_video = _decode_binding_rgb(final_path)
    approved_sampled_frames = len(approved_video) // VISUAL_BINDING_FRAME_BYTES
    final_sampled_frames = len(final_video) // VISUAL_BINDING_FRAME_BYTES
    correlation, lag, compared_frames, mean_absolute_error = _best_alignment(
        approved_video, final_video
    )
    unsigned = {
        "schema": VISUAL_BINDING_SCHEMA,
        "source_sha256": [_file_hash(path) for path in source_paths],
        "final_artifact_sha256": _file_hash(final_path),
        "sample_fps": VISUAL_BINDING_SAMPLE_FPS,
        "width": VISUAL_BINDING_WIDTH,
        "height": VISUAL_BINDING_HEIGHT,
        "channels": VISUAL_BINDING_CHANNELS,
        "approved_sampled_frames": approved_sampled_frames,
        "final_sampled_frames": final_sampled_frames,
        "max_frame_delta": VISUAL_BINDING_MAX_FRAME_DELTA,
        "compared_frames": compared_frames,
        "alignment_lag_frames": lag,
        "correlation": correlation,
        "minimum_correlation": threshold,
        "mean_absolute_error": mean_absolute_error,
        "maximum_mean_absolute_error": max_error,
        "accepted": correlation >= threshold and mean_absolute_error <= max_error,
    }
    evidence = VisualBindingEvidence(
        source_sha256=tuple(unsigned["source_sha256"]),
        final_artifact_sha256=unsigned["final_artifact_sha256"],
        sample_fps=VISUAL_BINDING_SAMPLE_FPS,
        width=VISUAL_BINDING_WIDTH,
        height=VISUAL_BINDING_HEIGHT,
        channels=VISUAL_BINDING_CHANNELS,
        approved_sampled_frames=approved_sampled_frames,
        final_sampled_frames=final_sampled_frames,
        max_frame_delta=VISUAL_BINDING_MAX_FRAME_DELTA,
        compared_frames=compared_frames,
        alignment_lag_frames=lag,
        correlation=correlation,
        minimum_correlation=threshold,
        mean_absolute_error=mean_absolute_error,
        maximum_mean_absolute_error=max_error,
        evidence_sha256=_evidence_hash(unsigned),
    )
    if not evidence.accepted:
        raise VisualBindingError(
            "final-film video does not match approved connected shots: "
            f"correlation={correlation:.6f}, required>={threshold:.6f}; "
            f"mean_absolute_error={mean_absolute_error:.6f}, allowed<={max_error:.6f}"
        )
    return evidence


__all__ = [
    "MAX_VISUAL_BINDING_MEAN_ABSOLUTE_ERROR",
    "MIN_VISUAL_BINDING_CORRELATION",
    "VISUAL_BINDING_CHANNELS",
    "VISUAL_BINDING_FRAME_BYTES",
    "VISUAL_BINDING_MAX_FRAME_DELTA",
    "VISUAL_BINDING_MAX_LAG_FRAMES",
    "VISUAL_BINDING_SAMPLE_FPS",
    "VISUAL_BINDING_SCHEMA",
    "VisualBindingError",
    "VisualBindingEvidence",
    "measure_visual_binding",
]
