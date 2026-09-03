"""Fail-closed assembly of production films from approved evidence-bound assets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .assembly import assemble
from .exceptions import AssemblyError
from .media_probe import MediaProbeError, probe_audio_signal, probe_media
from .validator import file_hash

PRODUCTION_EVIDENCE_SCHEMA = "cineos-production-film-evidence/0.4"
PRODUCTION_AUDIO_SAMPLE_RATE_HZ = 48_000
MAX_AUDIO_DURATION_DELTA_SECONDS = 0.75
MIN_AUDIO_MEAN_VOLUME_DB = -80.0
MIN_AUDIO_MAX_VOLUME_DB = -60.0


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_bound_shot(
    record: Mapping[str, Any], *, index: int
) -> tuple[str, Path, str]:
    shot_id = str(record.get("shot_id") or "").strip()
    if not shot_id:
        raise AssemblyError(f"production shot {index} is missing shot_id")
    if record.get("accepted") is not True or record.get("decision") != "accept":
        raise AssemblyError(f"production shot {shot_id} is not QC-approved")
    if record.get("production_gpu_evidence") is not True:
        raise AssemblyError(f"production shot {shot_id} lacks real GPU evidence")

    output_path = Path(str(record.get("output_path") or "")).resolve()
    expected_hash = str(record.get("output_sha256") or "").strip().lower()
    if len(expected_hash) != 64:
        raise AssemblyError(
            f"production shot {shot_id} is missing a valid output SHA-256"
        )
    actual_hash = file_hash(output_path)
    if actual_hash != expected_hash:
        raise AssemblyError(
            f"production shot {shot_id} artifact hash does not match QC evidence"
        )

    evidence_hash = str(record.get("evidence_sha256") or "").strip().lower()
    if len(evidence_hash) != 64:
        raise AssemblyError(f"production shot {shot_id} is missing evidence SHA-256")
    return shot_id, output_path, evidence_hash


def _validate_audio_stream(
    media: Mapping[str, Any], *, expected_duration: float | None
) -> dict[str, Any]:
    streams = media.get("audio_streams") or []
    if len(streams) != 1 or not isinstance(streams[0], Mapping):
        raise AssemblyError("production final MP4 is missing audio stream evidence")
    stream = streams[0]

    try:
        sample_rate = int(stream.get("sample_rate_hz") or 0)
        channels = int(stream.get("channels") or 0)
        duration = float(stream.get("duration_seconds") or 0.0)
    except (TypeError, ValueError):
        sample_rate = channels = 0
        duration = 0.0

    if sample_rate != PRODUCTION_AUDIO_SAMPLE_RATE_HZ:
        raise AssemblyError(
            "production final MP4 audio must be encoded at exactly "
            f"{PRODUCTION_AUDIO_SAMPLE_RATE_HZ} Hz"
        )
    if channels <= 0:
        raise AssemblyError("production final MP4 audio must have a valid channel count")
    if duration <= 0:
        raise AssemblyError(
            "production final MP4 audio must expose a positive stream duration"
        )

    duration_delta: float | None = None
    if expected_duration is not None:
        duration_delta = abs(duration - expected_duration)
        tolerance = max(
            MAX_AUDIO_DURATION_DELTA_SECONDS,
            expected_duration * 0.02,
        )
        if duration_delta > tolerance:
            raise AssemblyError(
                "production final audio duration deviates from the approved visual "
                f"timeline ({duration:.3f}s vs {expected_duration:.3f}s; "
                f"tolerance {tolerance:.3f}s)"
            )

    return {
        "codec_name": str(stream.get("codec_name") or ""),
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "duration_seconds": duration,
        "duration_delta_seconds": duration_delta,
    }


def _validate_audio_signal(movie: Path) -> dict[str, float]:
    try:
        signal = probe_audio_signal(movie)
    except MediaProbeError as exc:
        raise AssemblyError(
            f"production final audio signal validation failed: {exc}"
        ) from exc

    mean_volume = float(signal.get("mean_volume_db", -120.0))
    max_volume = float(signal.get("max_volume_db", -120.0))
    if mean_volume < MIN_AUDIO_MEAN_VOLUME_DB or max_volume < MIN_AUDIO_MAX_VOLUME_DB:
        raise AssemblyError(
            "production final audio is effectively silent or below the minimum "
            "decoded-signal floor"
        )
    return {
        "mean_volume_db": mean_volume,
        "max_volume_db": max_volume,
        "minimum_mean_volume_db": MIN_AUDIO_MEAN_VOLUME_DB,
        "minimum_max_volume_db": MIN_AUDIO_MAX_VOLUME_DB,
    }


def _validate_final_media(
    movie: Path,
    *,
    audio_required: bool,
    expected_duration: float | None,
) -> dict[str, Any]:
    try:
        media = probe_media(movie)
    except MediaProbeError as exc:
        raise AssemblyError(f"production final media validation failed: {exc}") from exc

    format_names = {
        item.strip().lower()
        for item in str(media.get("format_name") or "").split(",")
        if item.strip()
    }
    if "mp4" not in format_names:
        raise AssemblyError("production final artifact is not an MP4 container")

    if int(media.get("video_stream_count") or 0) != 1:
        raise AssemblyError(
            "production final MP4 must contain exactly one video stream"
        )
    video_codecs = [
        str(item).strip().lower() for item in media.get("video_codecs") or []
    ]
    if video_codecs != ["h264"]:
        raise AssemblyError(
            "production final MP4 must contain exactly one H.264 video stream"
        )

    dimensions = media.get("video_dimensions") or []
    if len(dimensions) != 1 or not isinstance(dimensions[0], Mapping):
        raise AssemblyError("production final MP4 is missing valid video dimensions")
    try:
        width = int(dimensions[0].get("width") or 0)
        height = int(dimensions[0].get("height") or 0)
    except (TypeError, ValueError):
        width = height = 0
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise AssemblyError(
            "production final MP4 has invalid H.264/yuv420p video dimensions"
        )

    audio_count = int(media.get("audio_stream_count") or 0)
    audio_codecs = [
        str(item).strip().lower() for item in media.get("audio_codecs") or []
    ]
    if audio_required:
        if audio_count != 1:
            raise AssemblyError(
                "production final MP4 must contain exactly one approved audio stream"
            )
        if audio_codecs != ["aac"]:
            raise AssemblyError(
                "production final MP4 approved audio stream must be AAC"
            )
        media["production_audio_stream"] = _validate_audio_stream(
            media, expected_duration=expected_duration
        )
        media["production_audio_signal"] = _validate_audio_signal(movie)
    elif audio_count:
        raise AssemblyError(
            "production final MP4 contains audio even though no approved audio artifact "
            "was supplied"
        )

    duration = float(media.get("duration_seconds") or 0.0)
    if duration <= 0:
        raise AssemblyError("production final MP4 has no positive duration")
    if expected_duration is not None:
        tolerance = max(0.5, expected_duration * 0.02)
        if abs(duration - expected_duration) > tolerance:
            raise AssemblyError(
                "production final MP4 duration deviates from the approved visual timeline "
                f"({duration:.3f}s vs {expected_duration:.3f}s; "
                f"tolerance {tolerance:.3f}s)"
            )
    return dict(media)


def assemble_production_film(
    shot_evidence: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    durations: Sequence[float] | None = None,
    audio_path: str | Path | None = None,
    audio_sha256: str | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble and validate only exact artifacts approved by GPU/QC evidence.

    Every shot and optional audio mix is hash-bound before FFmpeg runs. The final
    MP4 is inspected and, when audio is required, the encoded AAC stream must have
    production sample rate, timeline coverage, and measurable decoded signal.
    Signal presence is an integrity check only; it is not evidence of dialogue
    intelligibility, semantic correctness, or lip synchronization.
    """
    if not 5 <= len(shot_evidence) <= 10:
        raise AssemblyError("production connected-film assembly requires 5 to 10 shots")
    if durations is not None and len(durations) != len(shot_evidence):
        raise AssemblyError("duration count does not match production shot count")
    if durations is not None and any(float(value) <= 0 for value in durations):
        raise AssemblyError("production shot durations must all be positive")

    bound: list[tuple[str, Path, str]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(shot_evidence):
        item = _require_bound_shot(record, index=index)
        if item[0] in seen_ids:
            raise AssemblyError(f"duplicate production shot_id: {item[0]}")
        seen_ids.add(item[0])
        bound.append(item)

    audio: dict[str, Any] | None = None
    audio_source: Path | None = None
    if audio_path is not None:
        if not audio_sha256 or len(audio_sha256.strip()) != 64:
            raise AssemblyError("production audio requires an explicit SHA-256")
        audio_source = Path(audio_path).resolve()
        actual_audio_hash = file_hash(audio_source)
        if actual_audio_hash != audio_sha256.strip().lower():
            raise AssemblyError(
                "production audio artifact hash does not match supplied evidence"
            )
        audio = {"path": str(audio_source), "sha256": actual_audio_hash}
    elif audio_sha256 is not None:
        raise AssemblyError("audio SHA-256 was supplied without an audio artifact")

    destination = Path(output).resolve()
    movie = assemble(
        [path for _, path, _ in bound],
        destination,
        durations=list(durations) if durations is not None else None,
        audio_path=audio_source,
    )
    expected_duration = (
        sum(float(value) for value in durations) if durations is not None else None
    )
    final_media = _validate_final_media(
        movie,
        audio_required=audio_source is not None,
        expected_duration=expected_duration,
    )
    final_hash = file_hash(movie)

    manifest: dict[str, Any] = {
        "schema": PRODUCTION_EVIDENCE_SCHEMA,
        "shot_count": len(bound),
        "shots": [
            {
                "index": index,
                "shot_id": shot_id,
                "output_path": str(path),
                "output_sha256": file_hash(path),
                "evidence_sha256": evidence_hash,
            }
            for index, (shot_id, path, evidence_hash) in enumerate(bound)
        ],
        "audio": audio,
        "final_mp4": str(movie.resolve()),
        "final_mp4_sha256": final_hash,
        "final_media": final_media,
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)

    target = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else destination.with_suffix(".production.json")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = ["PRODUCTION_EVIDENCE_SCHEMA", "assemble_production_film"]
