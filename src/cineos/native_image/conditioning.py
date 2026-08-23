"""CINEOS-native image conditioning research contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cineos.atlas.native_request import NativeShotRequest

NATIVE_IMAGE_PLAN_SCHEMA = "cineos-native-image-plan/0.1"


@dataclass(slots=True)
class NativeImageConditioningPlan:
    shot_id: str
    scene_id: str
    width: int
    height: int
    seed: int
    identity_tokens: list[dict[str, Any]]
    composition_tokens: dict[str, Any]
    environment_tokens: dict[str, Any]
    continuity_tokens: dict[str, Any]
    performance_tokens: dict[str, Any]
    schema: str = NATIVE_IMAGE_PLAN_SCHEMA
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("content_hash", None)
        return data

    def refresh_hash(self) -> str:
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        self.content_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return self.content_hash


def compile_native_image_plan(
    request: NativeShotRequest,
) -> NativeImageConditioningPlan:
    """Compile a native shot request into model-agnostic image conditioning tokens."""
    if not request.content_hash:
        request.refresh_hash()
    camera = request.camera
    resolution = tuple(camera.get("resolution", (1920, 1080)))
    if len(resolution) != 2:
        raise ValueError("native image conditioning requires width/height resolution")

    identities = []
    for character in request.characters:
        refs = list(character.get("approved_reference_ids", []))
        if not refs:
            raise ValueError(
                "character identity conditioning requires approved references"
            )
        overrides = dict(character.get("scene_specific_overrides", {}))
        identities.append(
            {
                "character_uuid": character.get("character_uuid"),
                "cinedna_profile_id": character.get("cinedna_profile_id"),
                "cinedna_profile_version": character.get("cinedna_profile_version"),
                "reference_ids": refs,
                "primary_reference_id": overrides.get("primary_reference_id", refs[0]),
                "identity_invariants": list(character.get("identity_invariants", [])),
                "face_constraints": dict(character.get("face_constraints", {})),
                "body_constraints": dict(character.get("body_constraints", {})),
            }
        )

    plan = NativeImageConditioningPlan(
        shot_id=request.shot_id,
        scene_id=request.scene_id,
        width=int(resolution[0]),
        height=int(resolution[1]),
        seed=request.deterministic_seed,
        identity_tokens=identities,
        composition_tokens={
            "shot_type": camera.get("shot_type", ""),
            "framing": camera.get("framing", ""),
            "lens": camera.get("lens", ""),
            "camera_position": camera.get("camera_position"),
            "focus_target": camera.get("focus_target"),
            "depth_of_field_intent": camera.get("depth_of_field_intent"),
            "aspect_ratio": camera.get("aspect_ratio", ""),
        },
        environment_tokens=dict(request.environment or {}),
        continuity_tokens=dict(request.continuity),
        performance_tokens=dict(request.performance),
        metadata={
            "source_native_request_hash": request.content_hash,
            "research_only": True,
            "model_binding": None,
        },
    )
    plan.refresh_hash()
    return plan
