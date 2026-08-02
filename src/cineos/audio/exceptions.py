"""Audio production errors."""


class AudioError(Exception):
    """Base error for renderer-independent audio production."""


class AudioValidationError(AudioError, ValueError):
    """An audio project or cue is invalid."""


class VoiceCastingError(AudioError):
    """An approved, stable voice assignment cannot be made."""


class ProviderCapabilityError(AudioError):
    """A provider cannot meet a requested production capability."""


class AudioMixError(AudioError):
    """An audio mix could not be produced safely."""
