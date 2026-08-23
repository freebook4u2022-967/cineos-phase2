"""CINEOS-native, renderer-independent shot request contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from cineos.conditioning import ConditioningPackage

NATIVE_SHOT_SCHEMA = "cineos-native-shot-request/0.1"


@dataclass(slots=True)
class NativeShotRequest:
    shot_id: str
    scene_id: str
    camera: dict[str, Any]
    characters: list[dict[str, Any]]
    environment: dict[str, Any] | None
    wardrobe: list[dict[str, Any]]
    props: list[dict[str, Any]]
    continuity: dict[str, Any]
    performance: dict[str, Any]
    approved_reference_ids: list[str]
    deterministic_seed: int
    renderer_requirements: dict[str, Any]
    schema: str = NATIVE_SHOT_SCHEMA
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("content_hash", None)
        return data

    def refresh_hash(self) -> str:
        payload = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        self.content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.content_hash

    def to_dict(self) -> dict[str, Any]:
        if not self.content_hash:
            self.refresh_hash()
        return asdict(self)


def compile_native_shot_request(package: ConditioningPackage) -> NativeShotRequest:
    """Compile an existing CINEOS ConditioningPackage into a native shot request."""
    if not package.character_conditioning and not package.approved_reference_ids:
        raise ValueError(
            "native shot request requires approved conditioning references"
        )

    request = NativeShotRequest(
        shot_id=package.shot_id,
        scene_id=package.scene_id,
        camera=asdict(package.camera_conditioning),
        characters=[asdict(item) for item in package.character_conditioning],
        environment=(
            asdict(package.environment_conditioning)
            if package.environment_conditioning is not None
            else None
        ),
        wardrobe=[asdict(item) for item in package.wardrobe_conditioning],
        props=[asdict(item) for item in package.prop_conditioning],
        continuity=asdict(package.continuity_constraints),
        performance={
            "performance_package_id": package.performance_package_id,
            "dialogue_timing": package.dialogue_timing,
            "facial_targets": package.facial_targets,
            "body_performance_tracks": package.body_performance_tracks,
            "gesture_tracks": package.gesture_tracks,
            "eye_lines": package.eye_lines,
            "capability_requirements": package.performance_capability_requirements,
        },
        approved_reference_ids=list(package.approved_reference_ids),
        deterministic_seed=package.deterministic_seed,
        renderer_requirements=asdict(package.renderer_capability_requirements),
        metadata={
            "source_conditioning_schema": package.schema_version,
            "source_conditioning_hash": package.content_hash,
            **dict(package.metadata),
        },
    )
    request.refresh_hash()
    return request
