"""Media-content verification and the Mission One assembly gate."""

from __future__ import annotations

import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any

from .config import ColabRenderConfig
from .exceptions import AssemblyBlockedError
from .result import RenderContentStatus, ShotRenderResult


def _probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,nb_read_frames,nb_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    return json.loads(subprocess.run(command, check=True, capture_output=True).stdout)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _sample_gray_frames(path: Path, width: int, height: int) -> list[bytes]:
    # fps=3/duration distributes three samples without needing timestamp arithmetic.
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "fps=3/duration,scale=64:64,format=gray",
            "-frames:v",
            "3",
            "-f",
            "rawvideo",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    sample_size = 64 * 64
    return [
        raw[i : i + sample_size]
        for i in range(0, len(raw), sample_size)
        if len(raw[i : i + sample_size]) == sample_size
    ]


def validate_rendered_shot(
    path: Path,
    *,
    shot_id: str = "",
    config: ColabRenderConfig | None = None,
) -> ShotRenderResult:
    """Decode and conservatively validate a rendered MP4 without requiring a GPU."""
    config = config or ColabRenderConfig()
    base = dict(shot_id=shot_id or path.stem, output_file=path.name, success=False)
    if not path.is_file() or path.stat().st_size < config.minimum_file_size:
        return ShotRenderResult(
            **base,
            content_status=RenderContentStatus.EMPTY_OUTPUT,
            file_size=path.stat().st_size if path.exists() else 0,
        )
    try:
        probe = _probe(path)
        stream = (probe.get("streams") or [{}])[0]
        duration = _number(stream.get("duration")) or _number(
            probe.get("format", {}).get("duration")
        )
        width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
        frame_count = int(
            _number(stream.get("nb_read_frames") or stream.get("nb_frames"))
        )
        metrics = dict(
            file_size=path.stat().st_size,
            duration=duration,
            frame_count=frame_count,
            width=width,
            height=height,
        )
        if duration <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
            return ShotRenderResult(
                **base, content_status=RenderContentStatus.EMPTY_OUTPUT, **metrics
            )
        frames = _sample_gray_frames(path, width, height)
        if not frames:
            return ShotRenderResult(
                **base, content_status=RenderContentStatus.DECODE_FAILURE, **metrics
            )
        pixels = [value for frame in frames for value in frame]
        mean = statistics.fmean(pixels)
        variance = statistics.pvariance(pixels)
        metrics.update(
            mean_luminance=round(mean, 4), luminance_variance=round(variance, 4)
        )
        if (
            mean < config.black_luminance_threshold
            and variance < config.black_variance_threshold
        ):
            status = RenderContentStatus.BLACK_FRAME_FAILURE
        else:
            deltas = [
                statistics.fmean(abs(a - b) for a, b in zip(left, right))
                for left, right in zip(frames, frames[1:])
            ]
            status = (
                RenderContentStatus.FROZEN_FRAME_FAILURE
                if len(frames) > 1
                and max(deltas, default=0) < config.frozen_frame_delta_threshold
                else RenderContentStatus.VALID
            )
        return ShotRenderResult(
            **base,
            success=status == RenderContentStatus.VALID,
            content_status=status,
            **metrics,
        )
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError):
        return ShotRenderResult(
            **base,
            content_status=RenderContentStatus.DECODE_FAILURE,
            file_size=path.stat().st_size,
        )


def verify_results(source: Path) -> dict[str, Any]:
    data = json.loads(source.read_text(encoding="utf-8"))
    expected = data.get("expected_shots") or [
        x.get("shot_id") for x in data.get("shots", [])
    ]
    shots = data.get("shots", [])
    valid_ids = {
        item.get("shot_id")
        for item in shots
        if item.get("success", True)
        and item.get("content_status", RenderContentStatus.VALID)
        == RenderContentStatus.VALID
    }
    missing = [shot_id for shot_id in expected if shot_id not in valid_ids]
    statuses = [item.get("content_status") for item in shots]
    valid = bool(shots) and not missing
    return {
        "valid": valid,
        "total_shots": len(expected),
        "valid_shots": len(expected) - len(missing),
        "failed_shots": len(missing),
        "black_frame_failures": statuses.count(RenderContentStatus.BLACK_FRAME_FAILURE),
        "frozen_frame_warnings": statuses.count(
            RenderContentStatus.FROZEN_FRAME_FAILURE
        ),
        "missing_shots": missing,
        "final_assembly_status": "ready" if valid else "blocked",
        "hardware_profile": data.get("hardware_profile", data.get("hardware", "")),
        "model_profile": data.get(
            "model_profile", {"model_id": data.get("model_id", "")}
        ),
        "total_render_time": data.get("render_time_seconds", 0),
        "limitations": [
            "Reference conditioning is unsupported",
            "Lip-sync requires measurement",
        ],
        "manual_review_requirement": any(
            status == RenderContentStatus.MANUAL_REVIEW_REQUIRED for status in statuses
        ),
    }


def assemble(render_dir: Path, output: Path, fps: int | None = None) -> Path:
    report = render_dir / "render-results.json"
    if report.exists() and not verify_results(report)["valid"]:
        raise AssemblyBlockedError(
            "assembly blocked: one or more mandatory shots are invalid"
        )
    shots = sorted(render_dir.glob("shot-*.mp4")) or sorted(render_dir.glob("*.mp4"))
    if len(shots) != 3:
        raise AssemblyBlockedError(f"expected three rendered shots, found {len(shots)}")
    if shutil.which("ffmpeg") is None:
        raise AssemblyBlockedError("FFmpeg is required for assembly")
    output.parent.mkdir(parents=True, exist_ok=True)
    listing = render_dir / "concat.txt"
    listing.write_text("".join(f"file '{path.resolve()}'\n" for path in shots))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    if not output.exists() or output.stat().st_size == 0:
        raise AssemblyBlockedError("FFmpeg produced an empty film")
    return output


# Compatibility name used by early reliability prototypes.
validate_shot = validate_rendered_shot
