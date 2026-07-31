"""Typed links between canonical assets."""

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class RelationshipType(StrEnum):
    CHARACTER_WARDROBE = "character-wardrobe"
    CHARACTER_PROP = "character-prop"
    CHARACTER_VEHICLE = "character-vehicle"
    SCENE_ENVIRONMENT = "scene-environment"
    STORYBOARD_SCENE = "storyboard-scene"


@dataclass(frozen=True, slots=True)
class AssetRelationship:
    source_id: UUID
    target_id: UUID
    relationship: str
