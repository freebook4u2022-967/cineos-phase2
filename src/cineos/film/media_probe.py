"""Fail-closed FFprobe/FFmpeg inspection for production film artifacts."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


class MediaProbeError(RuntimeError):
    """Raised when a media artifact cannot be inspected reliably."""


def _ffprobe() -> str:
    executable = shutil.which("ffprobe")
    if not executable:
        raise MediaProbeError(
            "FFprobe is unavailable; install ffmpeg/ffprobe for production validation"
        )
    return executable


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise MediaProbeError(
            "FFmpeg is unavailable; install ffmpeg for production audio validation"
        )
    return executable


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _duration(payload: dict[str, Any]) -> float:
    raw = (payload.get("format") or {}).get("duration")
    value = _positive_float(raw)
    if value is not None:
        return value

    durations = [
        item
        for stream in payload.get("streams") or []
        if (item := _positive_float(stream.get("duration"))) is not None
    ]
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
        (
            "format=duration,format_name:"
            "stream=index,codec_type,codec_name,duration,width,height,"
            "sample_rate,channels"
        ),
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
        "audio_streams": [
            {
                "codec_name": str(item.get("codec_name") or ""),
                "sample_rate_hz": _positive_int(item.get("sample_rate")),
                "channels": _positive_int(item.get("channels")),
                "duration_seconds": _positive_float(item.get("duration")),
            }
            for item in audio
        ],
    }


_VOLUME_RE = re.compile(
    r"(?P<kind>mean_volume|max_volume):\s*(?P<value>-?inf|-?\d+(?:\.\d+)?)\s*dB",
    re.IGNORECASE,
)


def probe_audio_signal(path: str | Path) -> dict[str, float]:
    """Decode the primary audio stream and return signal-level evidence.

    FFmpeg's ``volumedetect`` runs over the encoded final artifact, so a muxed AAC
    stream containing only digital silence cannot pass merely because the stream
    exists. Infinite-negative silence is normalized to -120 dB to keep manifests
    standards-compliant JSON.
    """
    source = Path(path).resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise MediaProbeError(f"missing or empty media artifact: {source}")

    command = [
        _ffmpeg(),
        "-nostdin",
        "-v",
        "info",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise MediaProbeError(
            f"FFmpeg audio signal inspection failed: {result.stderr.strip()}"
        )

    values: dict[str, float] = {}
    for match in _VOLUME_RE.finditer(result.stderr):
        raw = match.group("value").lower()
        value = -120.0 if raw in {"-inf", "inf"} else float(raw)
        values[match.group("kind").lower()] = value
    if "mean_volume" not in values or "max_volume" not in values:
        raise MediaProbeError("FFmpeg did not report complete audio signal evidence")
    return {
        "mean_volume_db": values["mean_volume"],
        "max_volume_db": values["max_volume"],
    }


__all__ = ["MediaProbeError", "probe_audio_signal", "probe_media"]
