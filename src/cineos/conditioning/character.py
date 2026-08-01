from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CharacterConditioning:
    character_uuid: str
    cinedna_profile_id: str
    cinedna_profile_version: str
    approved_reference_ids: list[str]
    identity_invariants: list[str] = field(default_factory=list)
    face_constraints: dict[str, Any] = field(default_factory=dict)
    body_constraints: dict[str, Any] = field(default_factory=dict)
    expression_target: Any = None
    motion_target: Any = None
    scene_specific_overrides: dict[str, Any] = field(default_factory=dict)
