import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .beat import PerformanceBeat
from .body import BodyPerformanceTrack
from .emotion import EmotionalArc, EmotionalState
from .eyeline import EyeLineTrack
from .facial import FacialKeyframe, FacialPerformanceTrack
from .gesture import GestureTrack
from .lipsync import LipSyncTrack
from .plan import PerformanceCapabilityRequirements, PerformancePlan


def to_dict(plan, include_hash=True):
    data = asdict(plan)
    if not include_hash:
        data.pop("content_hash", None)
    return data


def serialize(plan, include_hash=True):
    return json.dumps(
        to_dict(plan, include_hash),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def calculate_content_hash(plan):
    return hashlib.sha256(serialize(plan, False).encode()).hexdigest()


def from_dict(v):
    faces = []
    for x in v.get("facial_performance_tracks", []):
        x = dict(x)
        x["keyframes"] = [FacialKeyframe(**k) for k in x.get("keyframes", [])]
        faces.append(FacialPerformanceTrack(**x))
    arc = dict(v.get("emotional_arc", {}))
    arc["states"] = [EmotionalState(**s) for s in arc.get("states", [])]
    return PerformancePlan(
        shot_id=v["shot_id"],
        scene_id=v["scene_id"],
        character_ids=list(v.get("character_ids", [])),
        dialogue_cue_ids=list(v.get("dialogue_cue_ids", [])),
        performance_beats=[
            PerformanceBeat(**x) for x in v.get("performance_beats", [])
        ],
        facial_performance_tracks=faces,
        lip_sync_tracks=[LipSyncTrack(**x) for x in v.get("lip_sync_tracks", [])],
        gesture_tracks=[GestureTrack(**x) for x in v.get("gesture_tracks", [])],
        body_performance_tracks=[
            BodyPerformanceTrack(**x) for x in v.get("body_performance_tracks", [])
        ],
        eye_line_tracks=[EyeLineTrack(**x) for x in v.get("eye_line_tracks", [])],
        emotional_arc=EmotionalArc(**arc),
        continuity_inputs=dict(v.get("continuity_inputs", {})),
        continuity_outputs=dict(v.get("continuity_outputs", {})),
        renderer_capability_requirements=PerformanceCapabilityRequirements(
            **v.get("renderer_capability_requirements", {})
        ),
        performance_id=v.get("performance_id", ""),
        content_hash=v.get("content_hash", ""),
        metadata=dict(v.get("metadata", {})),
        schema_version=v.get("schema_version", "1.0"),
        lost_capabilities=list(v.get("lost_capabilities", [])),
    )


def deserialize(data):
    v = json.loads(data)
    if not isinstance(v, dict):
        raise ValueError("performance JSON must contain an object")
    return from_dict(v)


def save(plan, path):
    plan.content_hash = calculate_content_hash(plan)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(serialize(plan) + "\n", encoding="utf-8")
    return dest


def load(path):
    return deserialize(Path(path).read_text(encoding="utf-8"))
