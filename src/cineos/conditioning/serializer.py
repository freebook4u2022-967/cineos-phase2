"""Canonical JSON encoding for conditioning packages."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .camera import CameraConditioning
from .character import CharacterConditioning
from .continuity import ContinuityConditioning
from .environment import EnvironmentConditioning
from .package import ConditioningPackage, RendererCapabilityRequirements
from .props import PropConditioning
from .wardrobe import WardrobeConditioning


def package_to_dict(
    package: ConditioningPackage, *, include_hash: bool = True
) -> dict[str, Any]:
    value = asdict(package)
    if not include_hash:
        value.pop("content_hash", None)
    return value


def calculate_content_hash(package: ConditioningPackage) -> str:
    return hashlib.sha256(serialize(package, include_hash=False).encode()).hexdigest()


def serialize(package: ConditioningPackage, *, include_hash: bool = True) -> str:
    return json.dumps(
        package_to_dict(package, include_hash=include_hash),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def package_from_dict(value: Mapping[str, Any]) -> ConditioningPackage:
    environment = value.get("environment_conditioning")
    camera = dict(value.get("camera_conditioning", {}))
    if "resolution" in camera:
        camera["resolution"] = tuple(camera["resolution"])
    requirements = dict(value.get("renderer_capability_requirements", {}))
    if "supported_resolution" in requirements:
        requirements["supported_resolution"] = tuple(
            requirements["supported_resolution"]
        )
    return ConditioningPackage(
        schema_version=str(value.get("schema_version", "")),
        shot_id=str(value["shot_id"]),
        scene_id=str(value["scene_id"]),
        character_conditioning=[
            CharacterConditioning(**item)
            for item in value.get("character_conditioning", [])
        ],
        environment_conditioning=(
            EnvironmentConditioning(**environment) if environment else None
        ),
        wardrobe_conditioning=[
            WardrobeConditioning(**item)
            for item in value.get("wardrobe_conditioning", [])
        ],
        prop_conditioning=[
            PropConditioning(**item) for item in value.get("prop_conditioning", [])
        ],
        camera_conditioning=CameraConditioning(**camera),
        continuity_constraints=ContinuityConditioning(
            **value.get("continuity_constraints", {})
        ),
        approved_reference_ids=list(value.get("approved_reference_ids", [])),
        renderer_capability_requirements=RendererCapabilityRequirements(**requirements),
        deterministic_seed=int(value["deterministic_seed"]),
        content_hash=str(value.get("content_hash", "")),
        metadata=dict(value.get("metadata", {})),
    )


def deserialize(data: str | bytes | bytearray) -> ConditioningPackage:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("conditioning JSON must contain an object")
    return package_from_dict(value)


def save(package: ConditioningPackage, path: str | Path) -> Path:
    package.content_hash = calculate_content_hash(package)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialize(package) + "\n", encoding="utf-8")
    return destination


def load(path: str | Path) -> ConditioningPackage:
    return deserialize(Path(path).read_text(encoding="utf-8"))
