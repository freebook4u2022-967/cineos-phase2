"""Renderer-independent facial identity descriptors."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class FaceProfile:
    reference_asset_ids: list[str] = field(default_factory=list)
    facial_feature_descriptors: dict[str, str] = field(default_factory=dict)
    age_range: str = ""
    skin_tone_description: str = ""
    eye_description: str = ""
    hair_description: str = ""
    facial_marks: list[str] = field(default_factory=list)
    approved_expressions: list[str] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
