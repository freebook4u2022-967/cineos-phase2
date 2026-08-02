from dataclasses import dataclass, field

DEFAULT_VISEMES = {
    "sil": "rest",
    "AA": "open",
    "AE": "open",
    "AH": "open",
    "B": "closed",
    "M": "closed",
    "P": "closed",
    "F": "teeth-lip",
    "V": "teeth-lip",
    "L": "tongue",
    "OW": "round",
    "UW": "round",
}


def phonemes_to_visemes(timeline, mapping=None):
    table = {**DEFAULT_VISEMES, **(mapping or {})}
    return [
        {
            **item,
            "viseme": table.get(str(item.get("phoneme", "sil")).upper(), "generic"),
        }
        for item in timeline
    ]


@dataclass(slots=True)
class LipSyncTrack:
    character_id: str
    dialogue_cue_id: str
    phoneme_timeline: list[dict[str, object]] = field(default_factory=list)
    viseme_timeline: list[dict[str, object]] = field(default_factory=list)
    word_timing: list[dict[str, object]] = field(default_factory=list)
    mouth_open_values: list[dict[str, object]] = field(default_factory=list)
    jaw_motion: list[dict[str, object]] = field(default_factory=list)
    confidence: float = 0.0
    fallback_mode: str = "none"
    source_audio_hash: str = ""
    timing_offsets: dict[str, float] = field(default_factory=dict)
    locked: bool = False

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        if not self.viseme_timeline and self.phoneme_timeline:
            self.viseme_timeline = phonemes_to_visemes(self.phoneme_timeline)
