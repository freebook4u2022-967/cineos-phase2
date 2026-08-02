"""Approved voice identity descriptions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from uuid import uuid4


@dataclass(slots=True)
class VoiceProfile:
    character_id: str
    language: str
    voice_id: str = field(default_factory=lambda: str(uuid4()))
    accent: str = ""
    vocal_age_range: str = ""
    pitch_range: str = ""
    cadence: str = ""
    timbre_description: str = ""
    emotional_range: list[str] = field(default_factory=list)
    approved_reference_ids: list[str] = field(default_factory=list)
    provider_configuration_reference: str | None = None
    version: str = "1.0"
    rights_approved: bool = False

    @property
    def character_uuid(self) -> str:
        return self.character_id

    @property
    def content_hash(self) -> str:
        value = asdict(self)
        value.pop("voice_id", None)
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
