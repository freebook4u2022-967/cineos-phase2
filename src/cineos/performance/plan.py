from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .beat import PerformanceBeat
from .body import BodyPerformanceTrack
from .emotion import EmotionalArc
from .eyeline import EyeLineTrack
from .facial import FacialPerformanceTrack
from .gesture import GestureTrack
from .lipsync import LipSyncTrack

PERFORMANCE_SCHEMA_VERSION = "1.0"
CAPABILITIES = frozenset(
    {
        "facial-control",
        "viseme-control",
        "audio-driven-lip-sync",
        "pose-control",
        "motion-reference",
        "gesture-control",
        "gaze-control",
        "expression-reference",
        "multi-character-performance",
        "temporal-control",
    }
)


@dataclass(slots=True)
class PerformanceCapabilityRequirements:
    facial_control: bool = False
    viseme_control: bool = False
    audio_driven_lip_sync: bool = False
    pose_control: bool = False
    motion_reference: bool = False
    gesture_control: bool = False
    gaze_control: bool = False
    expression_reference: bool = False
    multi_character_performance: bool = False
    temporal_control: bool = True

    def required_features(self):
        return {
            name.replace("_", "-")
            for name, value in vars_from_slots(self).items()
            if value
        }


def vars_from_slots(obj):
    return {name: getattr(obj, name) for name in obj.__slots__}


@dataclass(slots=True)
class PerformancePlan:
    shot_id: str
    scene_id: str
    character_ids: list[str] = field(default_factory=list)
    dialogue_cue_ids: list[str] = field(default_factory=list)
    performance_beats: list[PerformanceBeat] = field(default_factory=list)
    facial_performance_tracks: list[FacialPerformanceTrack] = field(
        default_factory=list
    )
    lip_sync_tracks: list[LipSyncTrack] = field(default_factory=list)
    gesture_tracks: list[GestureTrack] = field(default_factory=list)
    body_performance_tracks: list[BodyPerformanceTrack] = field(default_factory=list)
    eye_line_tracks: list[EyeLineTrack] = field(default_factory=list)
    emotional_arc: EmotionalArc = field(default_factory=EmotionalArc)
    continuity_inputs: dict[str, Any] = field(default_factory=dict)
    continuity_outputs: dict[str, Any] = field(default_factory=dict)
    renderer_capability_requirements: PerformanceCapabilityRequirements = field(
        default_factory=PerformanceCapabilityRequirements
    )
    performance_id: str = field(default_factory=lambda: str(uuid4()))
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = PERFORMANCE_SCHEMA_VERSION
    lost_capabilities: list[str] = field(default_factory=list)

    @property
    def performance_uuid(self):
        return self.performance_id

    @property
    def beats(self):
        return self.performance_beats

    @property
    def facial_tracks(self):
        return self.facial_performance_tracks

    @property
    def body_tracks(self):
        return self.body_performance_tracks

    @property
    def eyeline_tracks(self):
        return self.eye_line_tracks
