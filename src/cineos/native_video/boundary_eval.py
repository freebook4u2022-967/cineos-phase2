"""Decode real final-film pixels around planned scene boundaries.

The pure boundary metrics in :mod:`cineos.native_video.final_eval` are deliberately
independent from any media tool. This module supplies the production evidence
adapter: it samples actual decoded grayscale pixels immediately before and after
planned edit boundaries in the assembled movie and feeds those pixels into the
same deterministic evaluator used by CI.

FFmpeg is used only as a decoder/sampler. It does not generate or modify visual
content. Sampling is fail-closed: missing media, unavailable FFmpeg, incomplete
frames, invalid timestamps, or duplicate/out-of-order boundary times abort the
quality gate instead of silently weakening final-film continuity validation.

Production sampling uses a short temporal window on both sides of an edit rather
than trusting one frame. The samples are concatenated in timeline order before
metric evaluation, preserving the dependency-free metric contract while making
transient corruption or continuity drift adjacent to the edit measurable.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .final_eval import (
    SceneBoundaryEvalPolicy,
    SceneBoundaryEvalReport,
    SceneBoundarySample,
    evaluate_scene_boundaries,
)


@dataclass(frozen=True, slots=True)
class SceneBoundaryPoint:
    """One planned edit boundary in final-film timeline seconds."""

    from_scene_id: str
    to_scene_id: str
    boundary_seconds: float
    transition: str = "cut"

    def __post_init__(self) -> None:
        if not self.from_scene_id or not self.to_scene_id:
            raise ValueError("scene boundary point requires non-empty scene IDs")
        if self.boundary_seconds <= 0.0:
            raise ValueError("boundary_seconds must be positive")
        if self.transition not in {"cut", "match", "fade"}:
            raise ValueError("transition must be one of: cut, match, fade")


@dataclass(slots=True)
class FFmpegSceneBoundaryEvaluator:
    """Measure assembled-film scene boundaries from real decoded frame evidence.

    ``sample_offset_seconds`` places the nearest outgoing sample before the edit
    and the nearest incoming sample after it. ``sample_count`` and
    ``sample_stride_seconds`` extend those points into short temporal windows so
    production QC cannot pass solely because one sampled frame happened to look
    healthy. Three samples per side remain inexpensive while covering roughly a
    tenth of a second around a 24 fps edit with the defaults.
    """

    sample_width: int = 32
    sample_height: int = 18
    sample_offset_seconds: float = 0.05
    sample_count: int = 3
    sample_stride_seconds: float = 0.04
    ffmpeg_binary: str = "ffmpeg"
    policy: SceneBoundaryEvalPolicy = SceneBoundaryEvalPolicy()

    def __post_init__(self) -> None:
        if self.sample_width <= 0 or self.sample_height <= 0:
            raise ValueError("sample dimensions must be positive")
        if self.sample_offset_seconds <= 0.0:
            raise ValueError("sample_offset_seconds must be positive")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.sample_stride_seconds <= 0.0:
            raise ValueError("sample_stride_seconds must be positive")

    @property
    def frame_size(self) -> int:
        return self.sample_width * self.sample_height

    @property
    def evidence_size(self) -> int:
        return self.frame_size * self.sample_count

    def _decode_frame(self, binary: str, source: Path, timestamp: float) -> bytes:
        if timestamp < 0.0:
            raise ValueError("sample timestamp must be non-negative")
        command = [
            binary,
            "-v",
            "error",
            "-ss",
            f"{timestamp:.6f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            (
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
        if len(payload) != self.frame_size:
            raise RuntimeError(
                "ffmpeg returned incomplete scene-boundary frame evidence: "
                f"expected {self.frame_size} bytes, got {len(payload)}"
            )
        return payload

    def _sample_outgoing(
        self, binary: str, source: Path, boundary_seconds: float
    ) -> bytes:
        nearest = boundary_seconds - self.sample_offset_seconds
        timestamps = [
            max(
                0.0,
                nearest - self.sample_stride_seconds * index,
            )
            for index in reversed(range(self.sample_count))
        ]
        return b"".join(
            self._decode_frame(binary, source, timestamp) for timestamp in timestamps
        )

    def _sample_incoming(
        self, binary: str, source: Path, boundary_seconds: float
    ) -> bytes:
        nearest = boundary_seconds + self.sample_offset_seconds
        timestamps = [
            nearest + self.sample_stride_seconds * index
            for index in range(self.sample_count)
        ]
        return b"".join(
            self._decode_frame(binary, source, timestamp) for timestamp in timestamps
        )

    @staticmethod
    def _validate_boundaries(boundaries: Sequence[SceneBoundaryPoint]) -> None:
        if not boundaries:
            raise ValueError("at least one scene boundary point is required")
        previous = -1.0
        seen_pairs: set[tuple[str, str]] = set()
        for boundary in boundaries:
            if boundary.boundary_seconds <= previous:
                raise ValueError(
                    "scene boundary timestamps must be strictly increasing"
                )
            pair = (boundary.from_scene_id, boundary.to_scene_id)
            if pair in seen_pairs:
                raise ValueError("duplicate scene boundary pair is not allowed")
            previous = boundary.boundary_seconds
            seen_pairs.add(pair)

    def evaluate(
        self,
        movie_path: str | Path,
        boundaries: Sequence[SceneBoundaryPoint],
    ) -> SceneBoundaryEvalReport:
        """Decode temporal windows on both sides of every planned edit and run QC."""

        source = Path(movie_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        self._validate_boundaries(boundaries)
        binary = shutil.which(self.ffmpeg_binary)
        if binary is None:
            raise RuntimeError(
                f"{self.ffmpeg_binary} is required for measured scene-boundary QC"
            )

        samples: list[SceneBoundarySample] = []
        for boundary in boundaries:
            outgoing = self._sample_outgoing(binary, source, boundary.boundary_seconds)
            incoming = self._sample_incoming(binary, source, boundary.boundary_seconds)
            if (
                len(outgoing) != self.evidence_size
                or len(incoming) != self.evidence_size
            ):
                raise RuntimeError("scene-boundary temporal evidence is incomplete")
            samples.append(
                SceneBoundarySample(
                    from_scene_id=boundary.from_scene_id,
                    to_scene_id=boundary.to_scene_id,
                    outgoing_frame=outgoing,
                    incoming_frame=incoming,
                    transition=boundary.transition,
                )
            )

        return evaluate_scene_boundaries(tuple(samples), self.policy)
