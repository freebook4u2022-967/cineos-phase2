"""Safe, deterministic FFmpeg timeline assembly."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .exceptions import AssemblyError
from .validator import file_hash


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise AssemblyError("FFmpeg is unavailable; install ffmpeg to assemble a film")
    return executable


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
    is supplied, the exact referenced audio artifact is added as a second FFmpeg
    input and explicitly mapped as the final audio stream. Source-shot audio can
    therefore never supersede the approved mix through FFmpeg's automatic stream
    selection. ``-shortest`` prevents a longer mix from extending the visual timeline
    unexpectedly. Production audio is normalized to the 48 kHz film/video delivery
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
