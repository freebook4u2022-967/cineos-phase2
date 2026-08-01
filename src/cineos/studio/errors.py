"""Studio-facing errors and actionable error classification."""

from enum import StrEnum


class ErrorKind(StrEnum):
    FFMPEG_MISSING = "FFmpeg is missing or unavailable"
    INVALID_PROJECT = "The project is invalid"
    MISSING_ASSET = "A required asset is missing"
    REFERENCE_REQUIRED = "An approved reference is required"
    UNSUPPORTED_RENDERER = "The selected renderer is unsupported"
    MODEL_MISSING = "The renderer model is not installed"
    INSUFFICIENT_VRAM = "The renderer has insufficient VRAM"
    RENDER_FAILED = "Rendering failed"
    VALIDATION_FAILED = "Validation failed"
    RECOVERY_EXHAUSTED = "Automatic recovery was exhausted"
    CORRUPT_OUTPUT = "The output is corrupt"
    PERMISSION_DENIED = "Permission was denied"


class StudioError(RuntimeError):
    """An error suitable for presentation in the Studio UI."""

    def __init__(self, message: str, *, kind: ErrorKind | None = None) -> None:
        self.kind = kind
        super().__init__(message)
