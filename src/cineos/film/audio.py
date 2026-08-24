"""Audio timeline descriptions and safe FFmpeg final-film mixing."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .exceptions import AssemblyError


@dataclass(frozen=True, slots=True)
class AudioTrack:
    path: Path
    kind: str = "dialogue"
    gain: float = 1.0
    start: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0


def available_tracks(tracks: list[AudioTrack] | None) -> list[AudioTrack]:
    """Return real tracks; an empty result explicitly means silent fallback."""
    return [track for track in tracks or [] if track.path.is_file()]


def mux_primary_audio(
    video: str | Path,
    tracks: list[AudioTrack] | None,
    output: str | Path,
) -> Path:
    """Mux the first available completed audio mix into a rendered film.

    FIRST FILM deliberately treats upstream dialogue/music mixing as one primary
    mix. Missing audio preserves the video unchanged instead of blocking film
    completion; production multi-track mixing can evolve independently.
    """
    source = Path(video)
    if not source.is_file():
        raise AssemblyError(f"video is unavailable: {source}")
    usable = available_tracks(tracks)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not usable:
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return destination
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise AssemblyError("FFmpeg is unavailable; install ffmpeg to mux film audio")
    track = usable[0]
    command = [
        ffmpeg,
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-i",
        str(track.path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode or not destination.is_file():
        raise AssemblyError(f"FFmpeg audio mux failed: {result.stderr.strip()}")
    return destination
