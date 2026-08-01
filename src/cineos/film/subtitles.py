"""Optional sidecar subtitle generation (never burned in by default)."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Cue:
    start: float
    end: float
    text: str


def _time(value: float, webvtt: bool) -> str:
    millis = round(value * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    seconds, millis = divmod(millis, 1000)
    separator = "." if webvtt else ","
    return f"{hours:02}:{minutes:02}:{seconds:02}{separator}{millis:03}"


def export(
    cues: list[Cue],
    destination: str | Path,
    *,
    language: str = "en",
    enabled: bool = True,
) -> Path | None:
    if not enabled:
        return None
    path = Path(destination)
    webvtt = path.suffix.lower() == ".vtt"
    lines = ["WEBVTT", f"Language: {language}", ""] if webvtt else []
    for index, cue in enumerate(cues, 1):
        if not webvtt:
            lines.append(str(index))
        lines.extend(
            [f"{_time(cue.start, webvtt)} --> {_time(cue.end, webvtt)}", cue.text, ""]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
