"""Validation orchestration and dependency-optional keyframe extraction."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import FakeValidatorBackend, ValidationStatus, ValidatorBackend
from .environment import EnvironmentValidator
from .exceptions import KeyframeExtractionError, ValidationError
from .identity import IdentityValidator
from .props import PropValidator
from .report import ValidationReport
from .serializer import content_hash
from .temporal import TemporalValidator
from .thresholds import ValidationThresholds
from .wardrobe import WardrobeValidator

ProgressCallback = Callable[[str, float], None]


@dataclass(slots=True)
class ValidationContext:
    conditioning: Any
    frames: list[Path]
    backend: ValidatorBackend
    thresholds: ValidationThresholds


def extract_keyframes(
    video: str | Path, destination: str | Path, *, fps: float = 1.0
) -> list[Path]:
    """Extract representative PNG frames with FFmpeg when it is installed."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise KeyframeExtractionError("FFmpeg is not available")
    source, target = Path(video), Path(destination)
    if not source.is_file():
        raise KeyframeExtractionError(f"render does not exist: {source}")
    target.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"fps={fps}",
        str(target / "frame-%06d.png"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise KeyframeExtractionError(
            completed.stderr.strip() or "FFmpeg extraction failed"
        )
    return sorted(target.glob("frame-*.png"))


class ValidationPipeline:
    """Run independent validation categories after a renderer completes."""

    def __init__(
        self,
        backend: ValidatorBackend | None = None,
        *,
        thresholds: ValidationThresholds | None = None,
        validators=None,
    ) -> None:
        self.backend = backend or FakeValidatorBackend()
        self.thresholds = thresholds or ValidationThresholds()
        self.validators = list(
            (
                IdentityValidator(),
                WardrobeValidator(),
                PropValidator(),
                EnvironmentValidator(),
                TemporalValidator(),
            )
            if validators is None
            else validators
        )

    def validate(
        self,
        render_path: str | Path,
        conditioning: Any,
        *,
        shot_id: str | None = None,
        scene_id: str | None = None,
        renderer_id: str = "unknown",
        frames: list[str | Path] | None = None,
        progress: ProgressCallback | None = None,
    ) -> ValidationReport:
        source = Path(render_path)
        if not source.exists():
            raise ValidationError(f"render does not exist: {source}")
        temporary = None
        if frames is None:
            temporary = tempfile.TemporaryDirectory(prefix="cineos-validation-")
            try:
                frame_paths = extract_keyframes(source, temporary.name)
            except KeyframeExtractionError:
                # Backends may validate the container directly without CV/FFmpeg.
                frame_paths = [source]
        else:
            frame_paths = [Path(frame) for frame in frames]
        context = ValidationContext(
            conditioning, frame_paths, self.backend, self.thresholds
        )
        results = []
        for index, validator in enumerate(self.validators):
            if progress:
                progress(validator.category, index / len(self.validators))
            results.append(validator.validate(context))
        if progress:
            progress("complete", 1.0)
        failures = [message for result in results for message in result.failures]
        warnings = [message for result in results for message in result.warnings]
        scores = [result.score for result in results if result.score is not None]
        score = sum(scores) / len(scores) if scores else None
        statuses = {result.status for result in results}
        if ValidationStatus.FAIL in statuses or (
            score is not None and score < self.thresholds.warning_threshold
        ):
            status = ValidationStatus.FAIL
        elif ValidationStatus.MANUAL_REVIEW_REQUIRED in statuses:
            status = ValidationStatus.MANUAL_REVIEW_REQUIRED
        elif statuses == {ValidationStatus.UNSUPPORTED}:
            status = ValidationStatus.UNSUPPORTED
        elif (
            warnings
            or ValidationStatus.UNSUPPORTED in statuses
            or (score is not None and score < self.thresholds.pass_threshold)
        ):
            status = ValidationStatus.PASS_WITH_WARNINGS
        else:
            status = ValidationStatus.PASS
        report = ValidationReport(
            shot_id=shot_id or getattr(conditioning, "shot_id", "unknown"),
            scene_id=scene_id or getattr(conditioning, "scene_id", "unknown"),
            renderer_id=renderer_id,
            overall_status=status,
            overall_score=score,
            results=results,
            warnings=warnings,
            failures=failures,
            rerender_recommendation=(
                "Rerender this shot after correcting failed continuity constraints."
                if status is ValidationStatus.FAIL
                else None
            ),
            metadata={
                "backend_id": self.backend.backend_id,
                "frame_count": len(frame_paths),
            },
        )
        report.content_hash = content_hash(report)
        if temporary:
            temporary.cleanup()
        return report

    def validate_render_result(
        self, render_result: Any, conditioning: Any, **options: Any
    ) -> ValidationReport:
        """Validate after completion and attach the report to a RenderResult."""
        report = self.validate(
            getattr(render_result, "output_mp4_path"),
            conditioning,
            shot_id=getattr(render_result, "shot_id", None),
            renderer_id=getattr(render_result, "renderer_id", "unknown"),
            **options,
        )
        metadata = getattr(render_result, "renderer_metadata", None)
        if not isinstance(metadata, dict):
            raise ValidationError("RenderResult does not expose mutable metadata")
        from .serializer import report_to_dict

        metadata["validation_report"] = report_to_dict(report)
        metadata["mark_for_rerender"] = report.should_rerender
        return report
