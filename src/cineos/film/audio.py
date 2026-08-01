"""Audio timeline descriptions and safe FFmpeg mixing."""

from dataclasses import dataclass
from pathlib import Path


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
