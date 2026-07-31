"""Canonical JSON serialization for CineDNA profiles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import UUID

from .body import BodyProfile
from .constraints import ContinuityConstraints
from .expression import ExpressionProfile
from .face import FaceProfile
from .motion import MotionProfile
from .profile import CharacterDNA
from .voice import VoiceProfile
from .wardrobe import WardrobeProfile


def profile_to_dict(
    profile: CharacterDNA, *, include_hash: bool = True
) -> dict[str, Any]:
    value = asdict(profile)
    value["character_uuid"] = str(profile.character_uuid)
    if not include_hash:
        value.pop("content_hash", None)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def calculate_content_hash(profile: CharacterDNA) -> str:
    payload = canonical_json(profile_to_dict(profile, include_hash=False))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def profile_from_dict(
    value: Mapping[str, Any], *, verify_hash: bool = True
) -> CharacterDNA:
    expressions_value = value.get("expression_profiles", {})
    if isinstance(expressions_value, list):
        expressions_value = {item["name"]: item for item in expressions_value}
    profile = CharacterDNA(
        character_uuid=UUID(
            str(value.get("character_uuid", value.get("character_id")))
        ),
        profile_version=str(value.get("profile_version", "1.0")),
        display_name=str(value["display_name"]),
        approved_reference_ids=list(value.get("approved_reference_ids", [])),
        face_profile=FaceProfile(
            **dict(value.get("face_profile", value.get("face", {})))
        ),
        body_profile=BodyProfile(
            **dict(value.get("body_profile", value.get("body", {})))
        ),
        wardrobe_profiles=[
            WardrobeProfile(**dict(item))
            for item in value.get("wardrobe_profiles", value.get("wardrobe", []))
        ],
        voice_profile=VoiceProfile(
            **dict(value.get("voice_profile", value.get("voice", {})))
        ),
        motion_profile=MotionProfile(
            **dict(value.get("motion_profile", value.get("motion", {})))
        ),
        expression_profiles={
            str(name): ExpressionProfile(**({"name": name} | dict(item)))
            for name, item in dict(expressions_value).items()
        },
        continuity_constraints=ContinuityConstraints(
            **dict(value.get("continuity_constraints", value.get("constraints", {})))
        ),
        metadata=dict(value.get("metadata", {})),
        content_hash=str(value.get("content_hash", "")),
    )
    calculated = calculate_content_hash(profile)
    if verify_hash and profile.content_hash and profile.content_hash != calculated:
        raise ValueError("CineDNA content hash does not match profile content")
    profile.content_hash = calculated
    return profile


def serialize(profile: CharacterDNA) -> str:
    profile.refresh_content_hash()
    return canonical_json(profile_to_dict(profile)) + "\n"


def deserialize(
    data: str | bytes | bytearray, *, verify_hash: bool = True
) -> CharacterDNA:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("CineDNA JSON must contain an object")
    return profile_from_dict(value, verify_hash=verify_hash)


def save(profile: CharacterDNA, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialize(profile), encoding="utf-8")
    return destination


def load(path: str | Path, *, verify_hash: bool = True) -> CharacterDNA:
    return deserialize(Path(path).read_text(encoding="utf-8"), verify_hash=verify_hash)
