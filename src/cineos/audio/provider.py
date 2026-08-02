"""Provider-neutral synthesis contracts and deterministic test provider."""

from __future__ import annotations

import math
import struct
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from .dialogue import DialogueCue
from .exceptions import ProviderCapabilityError
from .voice import VoiceProfile


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    supported_languages: frozenset[str] = frozenset()
    approved_voice_references: bool = False
    voice_cloning: bool = False
    emotional_control: bool = False
    duration_control: bool = False
    ssml: bool = False
    streaming: bool = False
    phoneme_timestamps: bool = False
    word_timestamps: bool = False
    commercial_use_metadata: bool = False

    def supports(self, language: str, required: set[str] | None = None) -> bool:
        if self.supported_languages and language not in self.supported_languages:
            return False
        return all(bool(getattr(self, item, False)) for item in required or set())


@dataclass(slots=True)
class SynthesisResult:
    cue_id: str
    audio_path: Path
    duration: float
    provider_id: str
    phonemes: list[dict[str, object]] = field(default_factory=list)
    words: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


class TextToSpeechProvider(ABC):
    provider_id: str
    capabilities: ProviderCapabilities

    def negotiate(self, language: str, required: set[str] | None = None) -> None:
        if not self.capabilities.supports(language, required):
            missing = sorted(required or [])
            raise ProviderCapabilityError(
                f"provider {self.provider_id} cannot satisfy {language}: {missing}"
            )

    @abstractmethod
    def synthesize(
        self, cue: DialogueCue, voice: VoiceProfile, output: Path
    ) -> SynthesisResult:
        """Synthesize one approved cue without exposing provider secrets."""


class VoiceCloningProvider(TextToSpeechProvider, ABC):
    """Provider that may clone only explicitly rights-approved references."""

    def validate_clone_rights(self, voice: VoiceProfile) -> None:
        if not voice.rights_approved or not voice.approved_reference_ids:
            raise ProviderCapabilityError(
                "voice cloning requires explicit approved rights"
            )


class LocalProvider(TextToSpeechProvider, ABC):
    """Marker contract for providers executed locally."""


class CloudProvider(TextToSpeechProvider, ABC):
    """Marker contract for remote providers; configuration stores references only."""


class FakeDeterministicProvider(LocalProvider):
    """Create deterministic PCM tones for tests and offline development."""

    provider_id = "fake-deterministic"
    capabilities = ProviderCapabilities(
        supported_languages=frozenset({"en", "en-US", "und"}),
        approved_voice_references=True,
        emotional_control=True,
        duration_control=True,
        phoneme_timestamps=True,
        word_timestamps=True,
        commercial_use_metadata=True,
    )

    def __init__(self, sample_rate: int = 48_000) -> None:
        self.sample_rate = sample_rate

    def synthesize(
        self, cue: DialogueCue, voice: VoiceProfile, output: Path
    ) -> SynthesisResult:
        self.negotiate(cue.language, {"duration_control"})
        if cue.approved_voice_profile_id not in (None, voice.voice_id):
            raise ProviderCapabilityError("cue and supplied voice assignment conflict")
        output.parent.mkdir(parents=True, exist_ok=True)
        frames = max(1, round(cue.target_duration * self.sample_rate))
        frequency = 180 + int(voice.content_hash[:4], 16) % 240
        with wave.open(str(output), "wb") as target:
            target.setparams((1, 2, self.sample_rate, frames, "NONE", "not compressed"))
            data = bytearray()
            for index in range(frames):
                sample = int(
                    5000 * math.sin(2 * math.pi * frequency * index / self.sample_rate)
                )
                data.extend(struct.pack("<h", sample))
            target.writeframes(data)
        words = cue.line_text.split()
        width = cue.target_duration / max(len(words), 1)
        word_timing = [
            {"word": word, "start": index * width, "end": (index + 1) * width}
            for index, word in enumerate(words)
        ]
        phonemes = [
            {"phoneme": word[0].lower(), "start": item["start"], "end": item["end"]}
            for word, item in zip(words, word_timing, strict=True)
            if word
        ]
        return SynthesisResult(
            cue.cue_id,
            output,
            cue.target_duration,
            self.provider_id,
            phonemes,
            word_timing,
        )


# Friendly alias used by integrations.
FakeProvider = FakeDeterministicProvider
