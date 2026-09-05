"""Safe, deterministic FFmpeg timeline assembly."""

from __future__ import annotations

import math
import shutil
import subprocess
from collections.abc import Mapping
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


def _paths_alias(left: Path, right: Path) -> bool:
    """Return true when two paths name the same filesystem object."""
    if left == right:
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _reject_output_collision(
    destination: Path,
    *,
    sources: list[Path],
    audio_source: Path | None = None,
) -> None:
    """Protect evidence-bound inputs from destructive in-place assembly."""
    protected_inputs = set(sources)
    if audio_source is not None:
        protected_inputs.add(audio_source)
    if any(_paths_alias(destination, source) for source in protected_inputs):
        raise AssemblyError(
            "assembly output must be distinct from every source video and approved audio "
            "artifact"
        )


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
    if not isinstance(stream, Mapping):
        raise AssemblyError("approved audio artifact is missing valid stream evidence")

    codec_name = str(stream.get("codec_name") or "").strip().lower()
    if not codec_name:
        raise AssemblyError("approved audio artifact is missing codec evidence")
    try:
        sample_rate = int(stream.get("sample_rate_hz") or 0)
        channels = int(stream.get("channels") or 0)
    except (TypeError, ValueError):
        sample_rate = channels = 0
    if sample_rate <= 0:
        raise AssemblyError("approved audio artifact has no valid sample-rate evidence")
    if channels <= 0:
        raise AssemblyError(
            "approved audio artifact has no valid channel-count evidence"
        )

    try:
        duration = float(
            stream.get("duration_seconds") or media.get("duration_seconds") or 0.0
        )
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        raise AssemblyError("approved audio artifact has no finite positive duration")

    duration_shortfall: float | None = None
    if expected_duration is not None:
        if not math.isfinite(expected_duration) or expected_duration <= 0:
            raise AssemblyError(
                "approved visual timeline has no finite positive duration"
            )
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
        "codec_name": codec_name,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "duration_seconds": duration,
        "duration_shortfall_seconds": duration_shortfall,
    }


def _visual_timeline_duration(
    sources: list[Path],
    durations: list[float] | None,
    *,
    crossfade: float = 0.0,
) -> float:
    """Resolve the visual timeline length independently of an optional soundtrack."""
    if durations is not None:
        total = sum(durations)
    else:
        total = 0.0
        for source in sources:
            try:
                media = probe_media(source)
            except MediaProbeError as exc:
                raise AssemblyError(
                    f"visual timeline preflight failed for {source}: {exc}"
                ) from exc
            try:
                duration = float(media.get("duration_seconds") or 0.0)
            except (TypeError, ValueError):
                duration = 0.0
            if not math.isfinite(duration) or duration <= 0:
                raise AssemblyError(
                    f"visual timeline source has no valid positive duration: {source}"
                )
            total += duration

    total -= crossfade * max(0, len(sources) - 1)
    if not math.isfinite(total) or total <= 0:
        raise AssemblyError("visual timeline has no valid positive duration")
    return total


def _explicit_trim_filter(durations: list[float], *, crossfade: float = 0.0) -> str:
    """Build a decoded-frame trim graph for an explicit production timeline."""
    chains: list[str] = []
    labels: list[str] = []
    for index, duration in enumerate(durations):
        label = f"v{index}"
        chains.append(
            f"[{index}:v:0]trim=start=0:duration={duration:.6f},"
            f"setpts=PTS-STARTPTS[{label}]"
        )
        labels.append(label)

    if crossfade == 0.0:
        inputs = "".join(f"[{label}]" for label in labels)
        chains.append(inputs + f"concat=n={len(durations)}:v=1:a=0[filmv]")
        return ";".join(chains)

    previous = labels[0]
    cumulative_duration = durations[0]
    for index in range(1, len(durations)):
        output = "filmv" if index == len(durations) - 1 else f"xf{index}"
        offset = cumulative_duration - crossfade
        chains.append(
            f"[{previous}][{labels[index]}]xfade=transition=fade:"
            f"duration={crossfade:.6f}:offset={offset:.6f}[{output}]"
        )
        previous = output
        cumulative_duration += durations[index] - crossfade
    return ";".join(chains)


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
    is supplied, the exact referenced audio artifact is preflighted and explicitly
    mapped as the final audio stream. Source-shot audio can therefore never supersede
    the approved mix through FFmpeg's automatic stream selection.

    When explicit edit durations are supplied, every source becomes an independent
    FFmpeg input and the decoded video is hard-trimmed with ``trim`` before concat.
    A positive ``crossfade`` uses FFmpeg's decoded-frame ``xfade`` filter between
    those approved edits; it is intentionally supported only with explicit durations
    so transition offsets and the authoritative visual timeline are deterministic.
    The visual timeline duration is resolved before encoding and remains authoritative
    when audio is present: short approved audio is padded, long audio is trimmed, and
    audio can never terminate the video timeline. Production audio is normalized to
    the 48 kHz film/video delivery rate. The output path must never alias an input
    video or approved audio artifact, protecting evidence-bound assets from FFmpeg's
    destructive ``-y`` overwrite behavior. Filesystem aliases such as hard links are
    treated as collisions even when their path strings differ.
    """
    if not shots:
        raise AssemblyError("cannot assemble an empty timeline")
    sources = [Path(item).resolve() for item in shots]
    for source in sources:
        file_hash(source)
    destination = Path(output).resolve()
    _reject_output_collision(destination, sources=sources)

    if durations is not None and len(durations) != len(sources):
        raise AssemblyError("duration count does not match shot count")
    normalized_durations: list[float] | None = None
    if durations is not None:
        normalized_durations = [float(value) for value in durations]
        if any(
            not math.isfinite(value) or value <= 0 for value in normalized_durations
        ):
            raise AssemblyError("shot durations must all be finite and positive")

    crossfade = float(crossfade)
    if not math.isfinite(crossfade) or crossfade < 0:
        raise AssemblyError("crossfade must be finite and non-negative")
    if crossfade > 0:
        if normalized_durations is None:
            raise AssemblyError("crossfade requires explicit shot durations")
        if len(sources) < 2:
            raise AssemblyError("crossfade requires at least two shots")
        if any(crossfade >= duration for duration in normalized_durations):
            raise AssemblyError("crossfade must be shorter than every shot duration")

    audio_source: Path | None = None
    expected_duration: float | None = None
    if audio_path is not None:
        audio_source = Path(audio_path).resolve()
        file_hash(audio_source)
        _reject_output_collision(
            destination,
            sources=sources,
            audio_source=audio_source,
        )
        if crossfade > 0:
            expected_duration = _visual_timeline_duration(
                sources,
                normalized_durations,
                crossfade=crossfade,
            )
        else:
            expected_duration = _visual_timeline_duration(
                sources,
                normalized_durations,
            )
        _preflight_audio(audio_source, expected_duration=expected_duration)

    destination.parent.mkdir(parents=True, exist_ok=True)

    command = [_ffmpeg(), "-nostdin", "-y"]
    if normalized_durations is not None:
        for source in sources:
            command.extend(["-i", str(source)])
        if audio_source is not None:
            command.extend(["-i", str(audio_source)])
        if crossfade > 0:
            trim_filter = _explicit_trim_filter(
                normalized_durations,
                crossfade=crossfade,
            )
        else:
            trim_filter = _explicit_trim_filter(normalized_durations)
        command.extend(
            [
                "-filter_complex",
                trim_filter,
                "-map",
                "[filmv]",
            ]
        )
        if audio_source is not None:
            command.extend(["-map", f"{len(sources)}:a:0"])
        else:
            command.append("-an")
    else:
        manifest = destination.with_suffix(".concat.txt")
        lines: list[str] = []
        for source in sources:
            escaped = str(source).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        command.extend(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest),
            ]
        )
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
        command.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
                "-af",
                "apad",
                "-t",
                f"{expected_duration:.6f}",
            ]
        )
    command.extend(["-movflags", "+faststart", str(destination)])

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode or not destination.is_file():
        raise AssemblyError(f"FFmpeg assembly failed: {result.stderr.strip()}")
    return destination
