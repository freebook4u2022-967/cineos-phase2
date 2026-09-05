"""Fail-closed FFprobe/FFmpeg inspection for production film artifacts."""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from fractions import Fraction
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
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_frame_rate(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw or raw.upper() == "N/A" or raw == "0/0":
        return None
    try:
        parsed = float(Fraction(raw))
    except (ValueError, ZeroDivisionError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _reject_nonfinite_duration_evidence(payload: dict[str, Any]) -> None:
    """Reject explicit NaN/Inf timing metadata instead of silently falling back.

    FFprobe may provide several independent duration fields. A missing or ``N/A``
    field can legitimately require another timing source, but an explicitly
    non-finite value is corrupt evidence and must not be hidden by decoded-frame
    fallback arithmetic.
    """

    containers = [payload.get("format") or {}, *(payload.get("streams") or [])]
    for item in containers:
        if not isinstance(item, dict) or "duration" not in item:
            continue
        raw = item.get("duration")
        normalized = str(raw or "").strip()
        if not normalized or normalized.upper() == "N/A":
            continue
        try:
            parsed = float(normalized)
        except ValueError:
            continue
        if not math.isfinite(parsed):
            raise MediaProbeError("FFprobe did not report a positive media duration")


def _validate_video_frame_evidence(stream: dict[str, Any]) -> None:
    """Reject decoded-frame counts that contradict the stream's own timeline.

    ``nb_read_frames`` comes from decoding while ``avg_frame_rate`` and stream duration
    describe the same video timeline. A gross disagreement between those independent
    FFprobe observations is stronger evidence of corruption, sparse-frame output, or
    unreliable probe metadata than a merely positive frame count. Small differences are
    tolerated for timestamp and duration rounding, including ordinary VFR material.
    """
    frame_count = _positive_int(stream.get("nb_read_frames"))
    frame_rate = _positive_frame_rate(stream.get("avg_frame_rate"))
    duration = _positive_float(stream.get("duration"))
    if frame_count is None or frame_rate is None or duration is None:
        return

    expected = duration * frame_rate
    tolerance = max(2.0, math.ceil(expected * 0.02))
    if abs(frame_count - expected) > tolerance:
        raise MediaProbeError(
            "decoded video frame count conflicts with the stream timeline: "
            f"count={frame_count}, duration={duration:.6f}s, "
            f"avg_frame_rate={frame_rate:.6f}fps, expected≈{expected:.2f}"
        )


def _duration(payload: dict[str, Any]) -> float:
    """Resolve authoritative media duration, preferring actual video when present.

    Production film assembly discards source-shot audio. For audiovisual source shots,
    container duration can therefore overstate usable picture duration when an embedded
    audio stream outlasts the video. Prefer positive video-stream duration whenever the
    artifact contains video. When stream duration is unavailable, decoded frame count
    divided by average frame rate is the next-strongest visual timing evidence. Audio-only
    artifacts retain the container/stream fallback.
    """
    video = [
        stream
        for stream in payload.get("streams") or []
        if stream.get("codec_type") == "video"
    ]
    video_durations = [
        item
        for stream in video
        if (item := _positive_float(stream.get("duration"))) is not None
    ]
    if video_durations:
        return max(video_durations)

    decoded_video_durations = [
        frame_count / frame_rate
        for stream in video
        if (frame_count := _positive_int(stream.get("nb_read_frames"))) is not None
        if (frame_rate := _positive_frame_rate(stream.get("avg_frame_rate")))
        is not None
    ]
    if decoded_video_durations:
        return max(decoded_video_durations)

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
    """Return normalized stream evidence for an encoded media artifact.

    Production inspection asks FFprobe to decode/count video frames rather than trusting
    container timestamps alone. The resulting frame counts let higher-level film gates
    detect dropped/duplicated-frame or retiming drift that can otherwise hide behind a
    plausible duration and average frame rate.
    """
    source = Path(path).resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise MediaProbeError(f"missing or empty media artifact: {source}")

    command = [
        _ffprobe(),
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "format=duration,format_name:"
            "stream=index,codec_type,codec_name,duration,width,height,avg_frame_rate,"
            "nb_read_frames,sample_rate,channels"
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
    _reject_nonfinite_duration_evidence(payload)
    video = [item for item in streams if item.get("codec_type") == "video"]
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    for stream in video:
        _validate_video_frame_evidence(stream)
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
        "video_frame_rates": [
            str(item.get("avg_frame_rate") or "").strip() for item in video
        ],
        "video_frame_counts": [
            _positive_int(item.get("nb_read_frames")) for item in video
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
