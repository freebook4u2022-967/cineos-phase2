"""Canonical renderer-independent audio production project."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from .dialogue import DialogueCue
from .lipsync import LipSyncMetadata
from .music import MusicCue


@dataclass(slots=True)
class MixSettings:
    dialogue_priority: bool = True
    ambience_gain: float = 0.6
    effects_gain: float = 0.8
    music_gain: float = 0.7
    music_ducking_db: float = -8.0
    normalization_target: float = -16.0
    peak_limit_db: float = -1.0


@dataclass(slots=True)
class AudioProject:
    film_package_id: str
    language: str = "en"
    sample_rate: int = 48_000
    channel_layout: str = "stereo"
    project_id: str = ""
    dialogue_tracks: list[DialogueCue] = field(default_factory=list)
    ambience_tracks: list[Any] = field(default_factory=list)
    effects_tracks: list[Any] = field(default_factory=list)
    music_tracks: list[MusicCue] = field(default_factory=list)
    subtitle_metadata: list[dict[str, Any]] = field(default_factory=list)
    lip_sync_metadata: list[LipSyncMetadata] = field(default_factory=list)
    mix_settings: MixSettings = field(default_factory=MixSettings)
    output_targets: list[str] = field(default_factory=list)
    voice_profiles: list[Any] = field(default_factory=list)
    voice_assignments: dict[str, str] = field(default_factory=dict)
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.project_id:
            seed = f"{self.film_package_id}:{self.language}:{self.version}"
            self.project_id = str(uuid5(NAMESPACE_URL, seed))
        if self.sample_rate <= 0:
            raise ValueError("sample rate must be positive")

    @property
    def project_uuid(self) -> str:
        return self.project_id

    @property
    def content_hash(self) -> str:
        value = asdict(self)
        value.pop("project_id", None)
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode()).hexdigest()
