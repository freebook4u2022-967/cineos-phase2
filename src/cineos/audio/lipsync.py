"""Lip-sync timing metadata for future renderer plugins (no visual rendering)."""

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class LipSyncMetadata:
    shot_id: str
    character_id: str
    dialogue_cue_id: str
    phoneme_timeline: list[dict[str, object]] = field(default_factory=list)
    word_timeline: list[dict[str, object]] = field(default_factory=list)
    viseme_timeline: list[dict[str, object]] = field(default_factory=list)
    source_provider: str = ""
    timing_confidence: float = 0.0
    fallback_mode: str = "none"

    @property
    def content_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
