"""Stable JSON serialization for audio projects and metadata."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .cue import AudioCue
from .dialogue import DialogueCue
from .lipsync import LipSyncMetadata
from .music import MusicCue
from .project import AudioProject, MixSettings
from .voice import VoiceProfile


def project_to_dict(project: AudioProject) -> dict[str, Any]:
    value = asdict(project)
    value["content_hash"] = project.content_hash
    return value


def dumps(project: AudioProject, *, indent: int | None = 2) -> str:
    return json.dumps(
        project_to_dict(project), sort_keys=True, indent=indent, default=str
    )


def save(project: AudioProject, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps(project) + "\n", encoding="utf-8")
    return target


def project_from_dict(value: dict[str, Any]) -> AudioProject:
    data = dict(value)
    data.pop("content_hash", None)
    data["dialogue_tracks"] = [
        DialogueCue(**item) for item in data.get("dialogue_tracks", [])
    ]
    data["ambience_tracks"] = [
        AudioCue(**item) for item in data.get("ambience_tracks", [])
    ]
    data["effects_tracks"] = [
        AudioCue(**item) for item in data.get("effects_tracks", [])
    ]
    data["music_tracks"] = [MusicCue(**item) for item in data.get("music_tracks", [])]
    data["lip_sync_metadata"] = [
        LipSyncMetadata(**item) for item in data.get("lip_sync_metadata", [])
    ]
    data["voice_profiles"] = [
        VoiceProfile(**item) for item in data.get("voice_profiles", [])
    ]
    data["mix_settings"] = MixSettings(**data.get("mix_settings", {}))
    return AudioProject(**data)


def loads(payload: str) -> AudioProject:
    return project_from_dict(json.loads(payload))


def load(path: str | Path) -> AudioProject:
    return loads(Path(path).read_text(encoding="utf-8"))
