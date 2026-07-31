"""Cross-shot character continuity rules."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ContinuityConstraints:
    immutable_facial_traits: list[str] = field(default_factory=list)
    immutable_body_traits: list[str] = field(default_factory=list)
    wardrobe_locks: dict[str, str] = field(default_factory=dict)
    prop_locks: dict[str, str] = field(default_factory=dict)
    hairstyle_locks: dict[str, str] = field(default_factory=dict)
    forbidden_changes: list[str] = field(default_factory=list)
    scene_specific_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
