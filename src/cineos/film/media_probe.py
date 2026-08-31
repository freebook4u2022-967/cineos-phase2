"""Fail-closed FFprobe inspection for production film artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


class MediaProbeError(RuntimeError):
    """Raised when a media artifact cannot be inspected reliably."""


def _ffprobe() -> str:
    executable = shutil.which("ffprobe")
    if not executable:
        raise MediaProbeError("FFprobe is unavailable; install ffmpeg/ffprobe for production validation")
    return executable


def _duration(payload: dict[str, Any]) -> float:
    raw = (payload.get("format") or {}).get("duration")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.0
    if value > 0:
        return value

    durations: list[float] = []
    for stream in payload.get("streams") or []:
        try:
            item = float(stream.get("duration"))
        except (TypeError, ValueError, AttributeError):
            continue
        if item > 0:
            durations.append(item)
    if not durations:
        raise MediaProbeError("FFprobe did not report a positive media duration")
    return max(durations)


def probe_media(path: str | Path) -> dict[str, Any]:
    """Return normalized stream evidence for an encoded media artifact."""
    source = Path(path).resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise MediaProbeError(f"missing or empty media artifact: {source}")

    command = [
        _ffprobe(),
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name:stream=index,codec_type,codec_name,duration,width,height,sample_rate,channels",
        "-of",
        "json",
        str(source),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise MediaProbeError(f"FFprobe failed: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError("FFprobe returned malformed JSON") from exc

    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise MediaProbeError("FFprobe response is missing stream metadata")
    video = [item for item in streams if item.get("codec_type") == "video"]
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    return {
        "duration_seconds": _duration(payload),
        "format_name": str((payload.get("format") or {}).get("format_name") or ""),
        "video_stream_count": len(video),
        "audio_stream_count": len(audio),
        "video_codecs": [str(item.get("codec_name") or "") for item in video],
        "audio_codecs": [str(item.get("codec_name") or "") for item in audio],
        "video_dimensions": [
            {"width": item.get("width"), "height": item.get("height")} for item in video
        ],
    }


__all__ = ["MediaProbeError", "probe_media"]
