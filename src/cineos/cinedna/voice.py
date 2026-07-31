"""Voice identity descriptions (not voice models or embeddings)."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class VoiceProfile:
    language: str = ""
    accent: str = ""
    vocal_age: str = ""
    pitch_range: str = ""
    cadence: str = ""
    emotional_range: list[str] = field(default_factory=list)
    approved_voice_reference_ids: list[str] = field(default_factory=list)
