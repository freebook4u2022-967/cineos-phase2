"""Scene-level directing plan."""

from dataclasses import dataclass, field

from .continuity import ContinuityState
from .pacing import PacingPlan


@dataclass(slots=True)
class ScenePlan:
    scene_id: str
    title: str
    narrative_purpose: str
    dramatic_beat: str
    location_asset_id: str
    participating_character_ids: list[str]
    required_props: list[str] = field(default_factory=list)
    wardrobe_state: dict[str, str] = field(default_factory=dict)
    emotional_state: dict[str, str] = field(default_factory=dict)
    start_condition: str = ""
    end_condition: str = ""
    estimated_duration: float = 0.0
    continuity_inputs: ContinuityState = field(default_factory=ContinuityState)
    continuity_outputs: ContinuityState = field(default_factory=ContinuityState)
    pacing: PacingPlan = field(default_factory=PacingPlan)
    rationale: str = ""
