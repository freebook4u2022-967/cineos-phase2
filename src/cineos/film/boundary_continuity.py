"""Artifact-bound visual continuity evidence for connected production shots.

This module measures decoded frame continuity at declared continuous edit boundaries.
It is deliberately not a semantic identity, anatomy, physics, or generative-model score.
Intentional cuts are recorded explicitly and are never mislabeled as continuity passes.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .exceptions import AssemblyError

CONTINUITY_EVIDENCE_SCHEMA = "cineos-boundary-continuity-evidence/0.1"
DEFAULT_BOUNDARY_SIMILARITY_THRESHOLD = 0.82
_BOUNDARY_SAMPLE_WIDTH = 64
_BOUNDARY_SAMPLE_HEIGHT = 36
_BOUNDARY_END_OFFSET_SECONDS = 0.05
_VALID_TRANSITIONS = frozenset({"continuous", "cut"})


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise AssemblyError(
            "FFmpeg is unavailable; production boundary continuity cannot be measured"
        )
    return executable


def _decode_luma_frame(movie: Path, *, timestamp_seconds: float) -> bytes:
    """Decode one low-resolution luma frame from the exact encoded artifact."""
    if not movie.is_file() or movie.stat().st_size == 0:
        raise AssemblyError(f"missing or empty continuity artifact: {movie}")
    if timestamp_seconds < 0:
        raise AssemblyError("continuity frame timestamp cannot be negative")

    command = [
        _ffmpeg(),
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{timestamp_seconds:.6f}",
        "-i",
        str(movie),
        "-map",
        "0:v:0",
        "-frames:v",
        "1",
        "-vf",
        f"scale={_BOUNDARY_SAMPLE_WIDTH}:{_BOUNDARY_SAMPLE_HEIGHT}:flags=area,format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise AssemblyError(f"FFmpeg continuity frame decode failed: {stderr}")

    expected = _BOUNDARY_SAMPLE_WIDTH * _BOUNDARY_SAMPLE_HEIGHT
    if len(result.stdout) != expected:
        raise AssemblyError(
            "FFmpeg continuity frame decode returned incomplete evidence "
            f"({len(result.stdout)} bytes, expected {expected})"
        )
    return bytes(result.stdout)


def _luma_similarity(left: bytes, right: bytes) -> float:
    if len(left) != len(right) or not left:
        raise AssemblyError("continuity frame evidence has incompatible dimensions")
    mean_absolute_delta = sum(
        abs(a - b) for a, b in zip(left, right, strict=True)
    ) / len(left)
    return max(0.0, min(1.0, 1.0 - mean_absolute_delta / 255.0))


def measure_connected_boundaries(
    shots: Sequence[str | Path],
    *,
    transitions: Sequence[str],
    durations: Sequence[float] | None = None,
    minimum_similarity: float = DEFAULT_BOUNDARY_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """Measure declared continuous boundaries and fail closed on visual discontinuity.

    ``transitions`` has one entry per boundary between adjacent shots. ``continuous``
    boundaries are measured from decoded pixels in the exact source artifacts. ``cut``
    boundaries are explicitly exempt and recorded as unmeasured intentional edits.

    Continuous production boundaries require approved edit durations so the previous
    shot is sampled immediately before its actual edit endpoint, never from unused tail
    footage or an unrelated first frame.
    """
    paths = [Path(item).resolve() for item in shots]
    if not 2 <= len(paths) <= 10:
        raise AssemblyError("boundary continuity measurement requires 2 to 10 shots")
    if len(transitions) != len(paths) - 1:
        raise AssemblyError(
            "transition count must equal connected shot count minus one"
        )
    if durations is not None and len(durations) != len(paths):
        raise AssemblyError("duration count does not match continuity shot count")
    if not 0.0 <= float(minimum_similarity) <= 1.0:
        raise AssemblyError("continuity similarity threshold must be between 0 and 1")

    normalized = [str(item).strip().lower() for item in transitions]
    invalid = [item for item in normalized if item not in _VALID_TRANSITIONS]
    if invalid:
        raise AssemblyError(
            "unsupported production transition mode: " + ", ".join(sorted(set(invalid)))
        )
    if "continuous" in normalized and durations is None:
        raise AssemblyError(
            "continuous production boundaries require approved edit durations"
        )

    edit_durations: list[float] | None = None
    if durations is not None:
        edit_durations = [float(value) for value in durations]
        if any(value <= 0 for value in edit_durations):
            raise AssemblyError("continuity edit durations must all be positive")

    boundaries: list[dict[str, Any]] = []
    for index, mode in enumerate(normalized):
        left_path = paths[index]
        right_path = paths[index + 1]
        item: dict[str, Any] = {
            "boundary_index": index,
            "from_path": str(left_path),
            "to_path": str(right_path),
            "transition": mode,
        }
        if mode == "cut":
            item.update(
                {
                    "measured": False,
                    "accepted": True,
                    "reason": "intentional-cut-explicitly-declared",
                }
            )
            boundaries.append(item)
            continue

        assert edit_durations is not None
        left_timestamp = max(
            0.0,
            edit_durations[index] - _BOUNDARY_END_OFFSET_SECONDS,
        )
        left = _decode_luma_frame(left_path, timestamp_seconds=left_timestamp)
        right = _decode_luma_frame(right_path, timestamp_seconds=0.0)
        similarity = _luma_similarity(left, right)
        item.update(
            {
                "measured": True,
                "accepted": similarity >= minimum_similarity,
                "metric": "decoded-luma-mean-absolute-similarity",
                "similarity": similarity,
                "minimum_similarity": float(minimum_similarity),
                "sample_size": {
                    "width": _BOUNDARY_SAMPLE_WIDTH,
                    "height": _BOUNDARY_SAMPLE_HEIGHT,
                },
                "timing_source": "approved-edit-endpoint",
                "from_timestamp_seconds": left_timestamp,
                "to_timestamp_seconds": 0.0,
                "from_frame_sha256": hashlib.sha256(left).hexdigest(),
                "to_frame_sha256": hashlib.sha256(right).hexdigest(),
            }
        )
        if not item["accepted"]:
            raise AssemblyError(
                "production continuous boundary failed decoded visual continuity: "
                f"boundary {index} similarity {similarity:.4f} is below "
                f"{minimum_similarity:.4f}"
            )
        boundaries.append(item)

    return {
        "schema": CONTINUITY_EVIDENCE_SCHEMA,
        "shot_count": len(paths),
        "boundary_count": len(boundaries),
        "minimum_similarity": float(minimum_similarity),
        "metric_scope": "low-frequency decoded luma composition only",
        "limitations": (
            "not semantic identity, anatomy, action, physics, or dialogue evidence"
        ),
        "boundaries": boundaries,
    }


__all__ = [
    "CONTINUITY_EVIDENCE_SCHEMA",
    "DEFAULT_BOUNDARY_SIMILARITY_THRESHOLD",
    "measure_connected_boundaries",
]
