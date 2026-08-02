"""Renderer-independent voice and audio production API."""

from .ambience import AmbiencePlanner
from .casting import VoiceAssignment, VoiceCasting
from .cue import AudioCue, CueType
from .dialogue import DialogueCue
from .effects import EffectsPlanner
from .exceptions import (
    AudioError,
    AudioMixError,
    AudioValidationError,
    ProviderCapabilityError,
    VoiceCastingError,
)
from .export import AudioExporter
from .lipsync import LipSyncMetadata
from .mixer import Mixer, MixInput
from .music import MusicCue, MusicSourceType
from .project import AudioProject, MixSettings
from .provider import (
    CloudProvider,
    FakeDeterministicProvider,
    FakeProvider,
    LocalProvider,
    ProviderCapabilities,
    SynthesisResult,
    TextToSpeechProvider,
    VoiceCloningProvider,
)
from .registry import ProviderRegistry
from .serializer import load, save
from .timeline import AudioTimeline
from .validator import AudioValidator, ValidationReport
from .voice import VoiceProfile

__all__ = [
    "AmbiencePlanner",
    "AudioCue",
    "AudioError",
    "AudioExporter",
    "AudioProject",
    "AudioMixError",
    "AudioTimeline",
    "AudioValidator",
    "AudioValidationError",
    "CloudProvider",
    "CueType",
    "DialogueCue",
    "EffectsPlanner",
    "FakeDeterministicProvider",
    "FakeProvider",
    "LipSyncMetadata",
    "LocalProvider",
    "MixInput",
    "MixSettings",
    "Mixer",
    "MusicCue",
    "MusicSourceType",
    "ProviderCapabilities",
    "ProviderCapabilityError",
    "ProviderRegistry",
    "SynthesisResult",
    "TextToSpeechProvider",
    "ValidationReport",
    "VoiceAssignment",
    "VoiceCasting",
    "VoiceCastingError",
    "VoiceCloningProvider",
    "VoiceProfile",
    "load",
    "save",
]
