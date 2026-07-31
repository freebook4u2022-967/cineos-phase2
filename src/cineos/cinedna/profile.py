"""The complete CineDNA v1 character identity value type."""

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from .body import BodyProfile
from .constraints import ContinuityConstraints
from .expression import ExpressionProfile
from .face import FaceProfile
from .motion import MotionProfile
from .voice import VoiceProfile
from .wardrobe import WardrobeProfile

CINEDNA_PROFILE_VERSION = "1.0"


@dataclass(slots=True)
class CharacterDNA:
    character_uuid: UUID
    display_name: str
    approved_reference_ids: list[str]
    face_profile: FaceProfile
    body_profile: BodyProfile
    profile_version: str = CINEDNA_PROFILE_VERSION
    wardrobe_profiles: list[WardrobeProfile] = field(default_factory=list)
    voice_profile: VoiceProfile = field(default_factory=VoiceProfile)
    motion_profile: MotionProfile = field(default_factory=MotionProfile)
    expression_profiles: dict[str, ExpressionProfile] = field(default_factory=dict)
    continuity_constraints: ContinuityConstraints = field(
        default_factory=ContinuityConstraints
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        self.character_uuid = UUID(str(self.character_uuid))

    @property
    def character_id(self) -> UUID:
        """Compatibility alias used by project and command APIs."""

        return self.character_uuid

    def refresh_content_hash(self) -> str:
        from .serializer import calculate_content_hash

        self.content_hash = calculate_content_hash(self)
        return self.content_hash
