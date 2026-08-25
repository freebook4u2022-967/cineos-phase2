"""Measured final-film audio integrity for production CINEOS builds.

The native film pipeline must not declare a movie production-ready merely because
its pictures pass QC.  This module inspects the encoded container and verifies that
required audio is present, decodable, technically usable, and aligned with the
picture duration.  FFprobe is used only as an inspector; no external generator is
involved in acceptance or repair.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class AudioIntegrityPolicy:
    """Technical acceptance thresholds for a final-film audio stream."""

    min_sample_rate_hz: int = 16000
    min_channels: int = 1
    max_duration_delta_seconds: float = 0.75

    def __post_init__(self) -> None:
        if self.min_sample_rate_hz <= 0:
            raise ValueError("min_sample_rate_hz must be positive")
        if self.min_channels <= 0:
            raise ValueError("min_channels must be positive")
        if (
            not math.isfinite(self.max_duration_delta_seconds)
            or self.max_duration_delta_seconds < 0.0
        ):
            raise ValueError(
                "max_duration_delta_seconds must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class AudioStreamEvidence:
    """Normalized evidence extracted from the final encoded container."""

    codec_name: str
    sample_rate_hz: int
    channels: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class AudioIntegrityReport:
    """Auditable final-film audio decision."""

    decision: str
    required: bool
    stream: AudioStreamEvidence | None
    expected_duration_seconds: float | None
    duration_delta_seconds: float | None
    directives: tuple[str, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.decision in {"accept", "warn"}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AudioInspector(Protocol):
    def inspect(self, movie_path: str | Path) -> AudioStreamEvidence | None: ...


@dataclass(slots=True)
class FFprobeAudioInspector:
    """Read the primary encoded audio stream using ffprobe JSON output."""

    ffprobe_binary: str | None = None

    def inspect(self, movie_path: str | Path) -> AudioStreamEvidence | None:
        source = Path(movie_path)
        if not source.is_file():
            raise FileNotFoundError(source)

        ffprobe = self.ffprobe_binary or shutil.which("ffprobe")
        if not ffprobe:
            raise RuntimeError(
                "ffprobe is unavailable; install ffmpeg/ffprobe for audio QC"
            )

        command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,duration:format=duration",
            "-of",
            "json",
            str(source),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(
                f"ffprobe audio inspection failed: {result.stderr.strip()}"
            )

        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "ffprobe returned invalid JSON for audio inspection"
            ) from exc

        streams = payload.get("streams") or []
        if not streams:
            return None
        stream = streams[0]

        raw_duration = stream.get("duration")
        if raw_duration in {None, "N/A", ""}:
            raw_duration = (payload.get("format") or {}).get("duration")

        try:
            sample_rate = int(stream.get("sample_rate") or 0)
            channels = int(stream.get("channels") or 0)
            duration = float(raw_duration or 0.0)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "ffprobe returned malformed audio stream metadata"
            ) from exc

        if not math.isfinite(duration) or duration <= 0.0:
            raise RuntimeError("final-film audio duration must be finite and positive")

        return AudioStreamEvidence(
            codec_name=str(stream.get("codec_name") or "unknown").strip() or "unknown",
            sample_rate_hz=sample_rate,
            channels=channels,
            duration_seconds=duration,
        )


@dataclass(slots=True)
class FinalFilmAudioIntegrityGate:
    """Fail-closed technical gate for required final-film audio."""

    policy: AudioIntegrityPolicy = AudioIntegrityPolicy()
    inspector: AudioInspector | None = None

    def __post_init__(self) -> None:
        if self.inspector is None:
            self.inspector = FFprobeAudioInspector()

    def evaluate(
        self,
        movie_path: str | Path,
        *,
        expected_duration_seconds: float | None = None,
        required: bool = True,
    ) -> AudioIntegrityReport:
        source = Path(movie_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        if expected_duration_seconds is not None and (
            not math.isfinite(expected_duration_seconds)
            or expected_duration_seconds <= 0.0
        ):
            raise ValueError("expected_duration_seconds must be finite and positive")
        if self.inspector is None:
            raise RuntimeError("audio inspector is not configured")

        stream = self.inspector.inspect(source)
        if stream is None:
            if required:
                return AudioIntegrityReport(
                    decision="reject",
                    required=True,
                    stream=None,
                    expected_duration_seconds=expected_duration_seconds,
                    duration_delta_seconds=None,
                    directives=(
                        "restore or render the required final-film audio stream",
                    ),
                )
            return AudioIntegrityReport(
                decision="accept",
                required=False,
                stream=None,
                expected_duration_seconds=expected_duration_seconds,
                duration_delta_seconds=None,
            )

        directives: list[str] = []
        decision = "accept"
        if stream.sample_rate_hz < self.policy.min_sample_rate_hz:
            decision = "reject"
            directives.append(
                f"raise final-film audio sample rate to at least {self.policy.min_sample_rate_hz} Hz"
            )
        if stream.channels < self.policy.min_channels:
            decision = "reject"
            directives.append(
                f"encode at least {self.policy.min_channels} final-film audio channel(s)"
            )

        duration_delta: float | None = None
        if expected_duration_seconds is not None:
            duration_delta = abs(stream.duration_seconds - expected_duration_seconds)
            if duration_delta > self.policy.max_duration_delta_seconds:
                decision = "reject"
                directives.append(
                    "realign final-film audio duration with the authored picture timeline"
                )

        return AudioIntegrityReport(
            decision=decision,
            required=required,
            stream=stream,
            expected_duration_seconds=expected_duration_seconds,
            duration_delta_seconds=duration_delta,
            directives=tuple(dict.fromkeys(directives)),
        )


__all__ = [
    "AudioIntegrityPolicy",
    "AudioIntegrityReport",
    "AudioInspector",
    "AudioStreamEvidence",
    "FFprobeAudioInspector",
    "FinalFilmAudioIntegrityGate",
]
