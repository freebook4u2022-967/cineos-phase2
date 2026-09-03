"""Safe, deterministic FFmpeg timeline assembly."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from .exceptions import AssemblyError
from .media_probe import MediaProbeError, probe_media
from .validator import file_hash

MAX_APPROVED_AUDIO_SHORTFALL_SECONDS = 0.75


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise AssemblyError("FFmpeg is unavailable; install ffmpeg to assemble a film")
    return executable


def _preflight_audio(
    source: Path,
    *,
    expected_duration: float | None,
) -> dict[str, Any]:
    """Verify the approved audio input before spending a production encode."""
    try:
        media = probe_media(source)
    except MediaProbeError as exc:
        raise AssemblyError(f"approved audio preflight failed: {exc}") from exc

    try:
        audio_stream_count = int(media.get("audio_stream_count") or 0)
    except (TypeError, ValueError):
        audio_stream_count = 0
    streams = media.get("audio_streams") or []
    if audio_stream_count != 1 or len(streams) != 1:
        raise AssemblyError(
            "approved audio artifact must contain exactly one audio stream"
        )

    stream = streams[0]
    if not isinstance(stream, dict):
        raise AssemblyError("approved audio artifact is missing valid stream evidence")
    try:
        duration = float(
            stream.get("duration_seconds") or media.get("duration_seconds") or 0.0
        )
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        raise AssemblyError("approved audio artifact has no positive duration")

    duration_shortfall: float | None = None
    if expected_duration is not None:
        duration_shortfall = expected_duration - duration
        tolerance = max(
            MAX_APPROVED_AUDIO_SHORTFALL_SECONDS,
            expected_duration * 0.02,
        )
        if duration_shortfall > tolerance:
            raise AssemblyError(
                "approved audio artifact cannot cover the requested visual timeline "
                f"({duration:.3f}s audio vs {expected_duration:.3f}s video; "
                f"tolerance {tolerance:.3f}s)"
            )

    return {
        "audio_stream_count": audio_stream_count,
        "duration_seconds": duration,
        "duration_shortfall_seconds": duration_shortfall,
    }


def assemble(
    shots: list[str | Path],
    output: str | Path,
    *,
    durations: list[float] | None = None,
    crossfade: float = 0.0,
    audio_path: str | Path | None = None,
) -> Path:
    """Concatenate ordered shots and optionally mux an approved audio mix.

    The default remains video-only for backwards compatibility. When ``audio_path``
    is supplied, the exact referenced audio artifact is preflighted and added as a
    second FFmpeg input, then explicitly mapped as the final audio stream. Source-shot
    audio can therefore never supersede the approved mix through FFmpeg's automatic
    stream selection. When explicit edit durations are supplied, audio coverage is
    verified before encoding so ``-shortest`` cannot silently truncate the requested
    visual timeline. Production audio is normalized to the 48 kHz film/video delivery
    rate so downstream evidence is deterministic across source mixes.
    """
    if not shots:
        raise AssemblyError("cannot assemble an empty timeline")
    sources = [Path(item).resolve() for item in shots]
    for source in sources:
        file_hash(source)
    if durations is not None and len(durations) != len(sources):
        raise AssemblyError("duration count does not match shot count")
    if crossfade < 0:
        raise AssemblyError("crossfade must not be negative")

    audio_source: Path | None = None
    if audio_path is not None:
        audio_source = Path(audio_path).resolve()
        file_hash(audio_source)
        expected_duration = (
            sum(float(value) for value in durations) if durations is not None else None
        )
        _preflight_audio(audio_source, expected_duration=expected_duration)

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = destination.with_suffix(".concat.txt")
    lines: list[str] = []
    for index, source in enumerate(sources):
        escaped = str(source).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
        if durations:
            lines.append(f"duration {durations[index]:.6f}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    command = [
        _ffmpeg(),
        "-nostdin",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest),
    ]
    if audio_source is not None:
        command.extend(
            [
                "-i",
                str(audio_source),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
        )
    else:
        command.append("-an")

    command.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if audio_source is not None:
        command.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest"])
    command.extend(["-movflags", "+faststart", str(destination)])

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode or not destination.is_file():
        raise AssemblyError(f"FFmpeg assembly failed: {result.stderr.strip()}")
    return destination
