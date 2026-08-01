"""Validation subsystem exceptions."""


class ValidationError(RuntimeError):
    """Raised when a validation request cannot be completed."""


class KeyframeExtractionError(ValidationError):
    """Raised when FFmpeg cannot extract frames from a render."""
