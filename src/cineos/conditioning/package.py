from dataclasses import dataclass, field
from typing import Any

from .camera import CameraConditioning
from .character import CharacterConditioning
from .continuity import ContinuityConditioning
from .environment import EnvironmentConditioning
from .props import PropConditioning
from .wardrobe import WardrobeConditioning

CONDITIONING_SCHEMA_VERSION = "1.0"


@dataclass(slots=True)
class RendererCapabilityRequirements:
    image_reference_support: bool = False
    multi_reference_support: bool = False
    face_identity_support: bool = False
    character_count: int = 0
    maximum_duration: float = 0.0
    supported_resolution: tuple[int, int] = (1920, 1080)
    supported_fps: float = 24.0
    control_image_support: bool = False
    motion_reference_support: bool = False


@dataclass(slots=True)
class ConditioningPackage:
    shot_id: str
    scene_id: str
    character_conditioning: list[CharacterConditioning]
    environment_conditioning: EnvironmentConditioning | None
    wardrobe_conditioning: list[WardrobeConditioning]
    prop_conditioning: list[PropConditioning]
    camera_conditioning: CameraConditioning
    continuity_constraints: ContinuityConditioning
    approved_reference_ids: list[str]
    renderer_capability_requirements: RendererCapabilityRequirements
    deterministic_seed: int
    schema_version: str = CONDITIONING_SCHEMA_VERSION
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Concise compatibility aliases.
    @property
    def characters(self):
        return self.character_conditioning

    @property
    def camera(self):
        return self.camera_conditioning
