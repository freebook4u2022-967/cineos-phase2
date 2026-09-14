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

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("audio track kind must not be empty")
        if self.gain < 0:
            raise ValueError("audio track gain must be non-negative")
        if self.start < 0:
            raise ValueError("audio track start must be non-negative")
        if self.fade_in < 0 or self.fade_out < 0:
            raise ValueError("audio track fades must be non-negative")


def available_tracks(tracks: list[AudioTrack] | None) -> list[AudioTrack]:
    """Return real tracks; an empty result explicitly means silent fallback."""
    return [track for track in tracks or [] if track.path.is_file()]


def _audio_filter_graph(tracks: list[AudioTrack]) -> str:
    """Build an FFmpeg filter graph for deterministic timeline-aware mixing."""
    chains: list[str] = []
    labels: list[str] = []
    for index, track in enumerate(tracks, start=1):
        filters = [f"volume={track.gain:.6f}"]
        if track.fade_in:
            filters.append(f"afade=t=in:st=0:d={track.fade_in:.6f}")
        if track.fade_out:
            # Reverse-fade-reverse applies the fade to the true end of a track
            # without requiring ffprobe duration discovery.
            filters.extend(
                [
                    "areverse",
                    f"afade=t=in:st=0:d={track.fade_out:.6f}",
                    "areverse",
                ]
            )
        if track.start:
            delay_ms = int(round(track.start * 1000.0))
            filters.append(f"adelay=delays={delay_ms}:all=1")
        label = f"a{index}"
        chains.append(f"[{index}:a:0]{','.join(filters)}[{label}]")
        labels.append(f"[{label}]")
    inputs = "".join(labels)
    chains.append(
        f"{inputs}amix=inputs={len(tracks)}:duration=longest:"
        "dropout_transition=0,alimiter=limit=0.98,apad[aout]"
    )
    return ";".join(chains)


def mux_audio_tracks(
    video: str | Path,
    tracks: list[AudioTrack] | None,
    output: str | Path,
) -> Path:
    """Mix available timeline tracks and mux them into a rendered film.

    Track gain, start offset and edge fades are applied in one FFmpeg graph. The
    final mix is padded and ``-shortest`` is used so short audio never truncates
    the picture; the video stream remains the authoritative film duration.
    Missing audio preserves the video unchanged.
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

    command = [ffmpeg, "-nostdin", "-y", "-i", str(source)]
    for track in usable:
        command.extend(["-i", str(track.path)])
    command.extend(
        [
            "-filter_complex",
            _audio_filter_graph(usable),
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode or not destination.is_file():
        raise AssemblyError(f"FFmpeg audio mix failed: {result.stderr.strip()}")
    return destination


def mux_primary_audio(
    video: str | Path,
    tracks: list[AudioTrack] | None,
    output: str | Path,
) -> Path:
    """Backward-compatible mux of the first available completed audio track."""
    usable = available_tracks(tracks)
    return mux_audio_tracks(video, usable[:1], output)
