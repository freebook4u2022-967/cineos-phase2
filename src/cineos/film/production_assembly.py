"""Fail-closed assembly of production films from approved evidence-bound assets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .assembly import assemble
from .exceptions import AssemblyError
from .media_probe import MediaProbeError, probe_audio_signal, probe_media
from .validator import file_hash

PRODUCTION_EVIDENCE_SCHEMA = "cineos-production-film-evidence/0.10"
PRODUCTION_AUDIO_SAMPLE_RATE_HZ = 48_000
MAX_AUDIO_DURATION_DELTA_SECONDS = 0.75
MAX_EDIT_DURATION_OVERRUN_SECONDS = 0.05
MAX_FINAL_FRAME_COUNT_DELTA = 1
MIN_AUDIO_MEAN_VOLUME_DB = -80.0
MIN_AUDIO_MAX_VOLUME_DB = -60.0


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_frame_rate(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        rate = Fraction(raw)
    except (ValueError, ZeroDivisionError):
        return None
    if rate <= 0:
        return None
    return f"{rate.numerator}/{rate.denominator}"


def _ceil_fraction(value: Fraction) -> int:
    """Return the mathematical ceiling of a non-negative fraction exactly."""
    if value < 0:
        raise ValueError("frame span cannot be negative")
    return (value.numerator + value.denominator - 1) // value.denominator


def _require_bound_shot(
    record: Mapping[str, Any], *, index: int
) -> tuple[str, Path, str, str]:
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
    return shot_id, output_path, actual_hash, evidence_hash


def _validate_bound_shot_media(shot_id: str, movie: Path) -> dict[str, Any]:
    try:
        media = probe_media(movie)
    except MediaProbeError as exc:
        raise AssemblyError(
            f"production shot {shot_id} media validation failed: {exc}"
        ) from exc

    format_names = {
        item.strip().lower()
        for item in str(media.get("format_name") or "").split(",")
        if item.strip()
    }
    if "mp4" not in format_names:
        raise AssemblyError(f"production shot {shot_id} is not an MP4 container")
    if int(media.get("video_stream_count") or 0) != 1:
        raise AssemblyError(
            f"production shot {shot_id} must contain exactly one video stream"
        )
    video_codecs = [
        str(item).strip().lower() for item in media.get("video_codecs") or []
    ]
    if video_codecs != ["h264"]:
        raise AssemblyError(
            f"production shot {shot_id} must contain exactly one H.264 video stream"
        )

    dimensions = media.get("video_dimensions") or []
    if len(dimensions) != 1 or not isinstance(dimensions[0], Mapping):
        raise AssemblyError(
            f"production shot {shot_id} is missing valid video dimensions"
        )
    try:
        width = int(dimensions[0].get("width") or 0)
        height = int(dimensions[0].get("height") or 0)
        duration = float(media.get("duration_seconds") or 0.0)
        audio_stream_count = int(media.get("audio_stream_count") or 0)
    except (TypeError, ValueError):
        width = height = 0
        duration = 0.0
        audio_stream_count = 0
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise AssemblyError(
            f"production shot {shot_id} has invalid H.264/yuv420p video dimensions"
        )
    if duration <= 0:
        raise AssemblyError(f"production shot {shot_id} has no positive duration")

    frame_rate: str | None = None
    if "video_frame_rates" in media:
        frame_rates = media.get("video_frame_rates") or []
        if len(frame_rates) != 1:
            raise AssemblyError(
                f"production shot {shot_id} is missing valid frame-rate evidence"
            )
        frame_rate = _normalize_frame_rate(frame_rates[0])
        if frame_rate is None:
            raise AssemblyError(
                f"production shot {shot_id} has invalid average frame-rate evidence"
            )

    decoded_frame_count: int | None = None
    if "video_frame_counts" in media:
        frame_counts = media.get("video_frame_counts") or []
        if len(frame_counts) != 1:
            raise AssemblyError(
                f"production shot {shot_id} is missing decoded frame-count evidence"
            )
        try:
            decoded_frame_count = int(frame_counts[0])
        except (TypeError, ValueError):
            decoded_frame_count = 0
        if decoded_frame_count <= 0:
            raise AssemblyError(
                f"production shot {shot_id} has invalid decoded frame-count evidence"
            )

    return {
        "format_name": str(media.get("format_name") or ""),
        "video_codec": "h264",
        "width": width,
        "height": height,
        "frame_rate": frame_rate,
        "decoded_frame_count": decoded_frame_count,
        "duration_seconds": duration,
        "audio_stream_count": audio_stream_count,
    }


def _validate_connected_shot_compatibility(
    bound: Sequence[tuple[str, Path, str, str]],
    shot_media: Mapping[str, Mapping[str, Any]],
    *,
    durations: Sequence[float] | None,
) -> dict[str, Any]:
    """Fail before FFmpeg when connected-shot media cannot form a truthful timeline."""
    first_shot_id = bound[0][0]
    expected_width = int(shot_media[first_shot_id]["width"])
    expected_height = int(shot_media[first_shot_id]["height"])
    expected_frame_rate = shot_media[first_shot_id].get("frame_rate")
    has_frame_rate_evidence = any(
        shot_media[shot_id].get("frame_rate") is not None for shot_id, _, _, _ in bound
    )
    if has_frame_rate_evidence and expected_frame_rate is None:
        raise AssemblyError(
            "production connected shots have incomplete frame-rate evidence"
        )

    edit_durations: list[float] | None = None
    if durations is not None:
        edit_durations = [float(value) for value in durations]

    for index, (shot_id, _, _, _) in enumerate(bound):
        media = shot_media[shot_id]
        width = int(media["width"])
        height = int(media["height"])
        if (width, height) != (expected_width, expected_height):
            raise AssemblyError(
                "production connected shots must use identical frame dimensions; "
                f"shot {shot_id} is {width}x{height}, expected "
                f"{expected_width}x{expected_height}"
            )

        frame_rate = media.get("frame_rate")
        if has_frame_rate_evidence:
            if frame_rate is None:
                raise AssemblyError(
                    "production connected shots have incomplete frame-rate evidence; "
                    f"shot {shot_id} has no valid average frame rate"
                )
            if frame_rate != expected_frame_rate:
                raise AssemblyError(
                    "production connected shots must use identical average frame rates; "
                    f"shot {shot_id} is {frame_rate}, expected {expected_frame_rate}"
                )

        if edit_durations is not None:
            requested = edit_durations[index]
            available = float(media["duration_seconds"])
            tolerance = max(
                MAX_EDIT_DURATION_OVERRUN_SECONDS,
                available * 0.01,
            )
            if requested > available + tolerance:
                raise AssemblyError(
                    f"production shot {shot_id} edit duration exceeds approved source "
                    f"duration ({requested:.3f}s requested vs {available:.3f}s available; "
                    f"tolerance {tolerance:.3f}s)"
                )

    return {
        "width": expected_width,
        "height": expected_height,
        "frame_rate": expected_frame_rate,
        "edit_durations_seconds": edit_durations,
    }


def _expected_timeline_frame_count(
    bound: Sequence[tuple[str, Path, str, str]],
    shot_media: Mapping[str, Mapping[str, Any]],
    *,
    frame_rate: str | None,
    durations: Sequence[float] | None,
) -> dict[str, Any]:
    """Derive decoded-frame expectations from the same semantics assembly uses.

    Explicit edit durations are hard-trimmed independently by FFmpeg. For CFR production
    sources, each trim retains frames whose timestamps occur before the requested end, so
    the exact per-shot expectation is ``ceil(duration * fps)`` rather than rounding the
    aggregate film duration. When the source probe exposes a decoded frame count, the trim
    expectation is capped at the source count. Untrimmed assembly uses observed decoded
    source counts directly and therefore does not infer frames from container duration.
    """
    if frame_rate is None:
        return {
            "mode": "unavailable-no-frame-rate",
            "expected_decoded_frame_count": None,
            "expected_per_shot_decoded_frame_counts": None,
        }

    rate = Fraction(frame_rate)
    per_shot: list[int] = []
    if durations is not None:
        for index, (shot_id, _, _, _) in enumerate(bound):
            requested = Fraction(str(durations[index]))
            expected = _ceil_fraction(requested * rate)
            source_count = shot_media[shot_id].get("decoded_frame_count")
            if source_count is not None:
                expected = min(expected, int(source_count))
            if expected <= 0:
                raise AssemblyError(
                    f"production shot {shot_id} edit implies no positive decoded frames"
                )
            per_shot.append(expected)
        mode = "per-shot-cfr-hard-trim"
    else:
        for shot_id, _, _, _ in bound:
            source_count = shot_media[shot_id].get("decoded_frame_count")
            if source_count is None:
                raise AssemblyError(
                    "production connected shots require decoded frame-count evidence "
                    "when assembling the full untrimmed source timeline"
                )
            per_shot.append(int(source_count))
        mode = "observed-source-decoded-frames"

    return {
        "mode": mode,
        "expected_decoded_frame_count": sum(per_shot),
        "expected_per_shot_decoded_frame_counts": per_shot,
    }


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
        raise AssemblyError(
            "production final MP4 audio must have a valid channel count"
        )
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
    expected_width: int,
    expected_height: int,
    expected_frame_rate: str | None,
    expected_frame_count: int | None,
    frame_count_mode: str,
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
    if (width, height) != (expected_width, expected_height):
        raise AssemblyError(
            "production final MP4 dimensions do not match the approved connected "
            f"timeline ({width}x{height} vs {expected_width}x{expected_height})"
        )

    final_frame_rate: str | None = None
    final_frame_count: int | None = None
    if expected_frame_rate is not None:
        frame_rates = media.get("video_frame_rates") or []
        if len(frame_rates) != 1:
            raise AssemblyError(
                "production final MP4 is missing average frame-rate evidence"
            )
        final_frame_rate = _normalize_frame_rate(frame_rates[0])
        if final_frame_rate is None:
            raise AssemblyError(
                "production final MP4 has invalid average frame-rate evidence"
            )
        if final_frame_rate != expected_frame_rate:
            raise AssemblyError(
                "production final MP4 average frame rate does not match the approved "
                f"connected timeline ({final_frame_rate} vs {expected_frame_rate})"
            )

        if expected_frame_count is not None:
            frame_counts = media.get("video_frame_counts") or []
            if len(frame_counts) != 1:
                raise AssemblyError(
                    "production final MP4 is missing decoded frame-count evidence"
                )
            try:
                final_frame_count = int(frame_counts[0])
            except (TypeError, ValueError):
                final_frame_count = 0
            if final_frame_count <= 0:
                raise AssemblyError(
                    "production final MP4 has invalid decoded frame-count evidence"
                )
            if expected_frame_count <= 0:
                raise AssemblyError(
                    "approved production timeline implies no positive decoded frame count"
                )
            delta = abs(final_frame_count - expected_frame_count)
            if delta > MAX_FINAL_FRAME_COUNT_DELTA:
                raise AssemblyError(
                    "production final MP4 decoded frame count does not match the approved "
                    f"connected timeline ({final_frame_count} vs {expected_frame_count}; "
                    f"tolerance {MAX_FINAL_FRAME_COUNT_DELTA} frame)"
                )

    media["production_video_timeline"] = {
        "width": width,
        "height": height,
        "frame_rate": final_frame_rate,
        "decoded_frame_count": final_frame_count,
        "expected_width": expected_width,
        "expected_height": expected_height,
        "expected_frame_rate": expected_frame_rate,
        "expected_decoded_frame_count": expected_frame_count,
        "expected_decoded_frame_count_mode": frame_count_mode,
        "decoded_frame_count_tolerance": MAX_FINAL_FRAME_COUNT_DELTA,
    }

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

    Every shot and optional audio mix is hash-bound before FFmpeg runs. Each bound shot
    is independently media-probed so a corrupt, mislabeled, or non-video artifact cannot
    reach final assembly merely because its hash matches metadata. Connected shots must
    use identical frame dimensions and average frame rates, and explicit edit durations
    cannot exceed the independently probed approved source footage. Source-shot audio is
    recorded as evidence but cannot supersede an approved external mix because assembly
    explicitly maps that mix. A connected production film must contain distinct rendered
    artifacts backed by distinct QC evidence, so one successful render cannot be relabeled
    to satisfy the 5-10-shot release gate. The final MP4 must preserve the approved frame
    geometry and normalized average frame rate. Decoded-frame validation mirrors assembly
    semantics: explicit CFR edit durations are quantified per hard-trimmed shot before
    concatenation, while untrimmed timelines bind directly to observed source frame counts.
    Source-shot decoded frame counts are evidence, not a duration-derived VFR assumption.
    When audio is required, the encoded AAC stream must have production sample rate,
    timeline coverage, and measurable decoded signal. Signal presence is an integrity check
    only; it is not evidence of dialogue intelligibility, semantic correctness, or lip
    synchronization.
    """
    if not 5 <= len(shot_evidence) <= 10:
        raise AssemblyError("production connected-film assembly requires 5 to 10 shots")
    if durations is not None and len(durations) != len(shot_evidence):
        raise AssemblyError("duration count does not match production shot count")
    if durations is not None and any(float(value) <= 0 for value in durations):
        raise AssemblyError("production shot durations must all be positive")

    bound: list[tuple[str, Path, str, str]] = []
    seen_ids: set[str] = set()
    seen_output_hashes: set[str] = set()
    seen_evidence_hashes: set[str] = set()
    for index, record in enumerate(shot_evidence):
        item = _require_bound_shot(record, index=index)
        shot_id, _, output_hash, evidence_hash = item
        if shot_id in seen_ids:
            raise AssemblyError(f"duplicate production shot_id: {shot_id}")
        if output_hash in seen_output_hashes:
            raise AssemblyError(
                f"production shot {shot_id} reuses a rendered artifact from another shot"
            )
        if evidence_hash in seen_evidence_hashes:
            raise AssemblyError(
                f"production shot {shot_id} reuses QC evidence from another shot"
            )
        seen_ids.add(shot_id)
        seen_output_hashes.add(output_hash)
        seen_evidence_hashes.add(evidence_hash)
        bound.append(item)

    shot_media = {
        shot_id: _validate_bound_shot_media(shot_id, path)
        for shot_id, path, _, _ in bound
    }
    timeline_compatibility = _validate_connected_shot_compatibility(
        bound,
        shot_media,
        durations=durations,
    )
    frame_count_expectation = _expected_timeline_frame_count(
        bound,
        shot_media,
        frame_rate=timeline_compatibility.get("frame_rate"),
        durations=durations,
    )

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
        [path for _, path, _, _ in bound],
        destination,
        durations=list(durations) if durations is not None else None,
        audio_path=audio_source,
    )
    if durations is not None:
        expected_duration = sum(float(value) for value in durations)
        timeline_source = "explicit-edit-durations"
    else:
        expected_duration = sum(
            float(shot_media[shot_id]["duration_seconds"]) for shot_id, _, _, _ in bound
        )
        timeline_source = "probed-source-shots"
    final_media = _validate_final_media(
        movie,
        audio_required=audio_source is not None,
        expected_duration=expected_duration,
        expected_width=int(timeline_compatibility["width"]),
        expected_height=int(timeline_compatibility["height"]),
        expected_frame_rate=timeline_compatibility.get("frame_rate"),
        expected_frame_count=frame_count_expectation["expected_decoded_frame_count"],
        frame_count_mode=str(frame_count_expectation["mode"]),
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
                "output_sha256": output_hash,
                "evidence_sha256": evidence_hash,
                "media": shot_media[shot_id],
            }
            for index, (shot_id, path, output_hash, evidence_hash) in enumerate(bound)
        ],
        "timeline": {
            "source": timeline_source,
            "expected_duration_seconds": expected_duration,
            "compatibility": timeline_compatibility,
            "frame_count_expectation": frame_count_expectation,
        },
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