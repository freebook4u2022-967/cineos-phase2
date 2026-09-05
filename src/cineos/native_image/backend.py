"""Research-only execution boundary for CINEOS native image generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .conditioning import NativeImageConditioningPlan


class NativeImageModel(Protocol):
    """Minimal future learned-model boundary owned by CINEOS."""

    def encode_identity(self, tokens: list[dict[str, Any]]) -> Any: ...

    def encode_scene(self, plan: NativeImageConditioningPlan) -> Any: ...

    def generate(self, *, identity_state: Any, scene_state: Any, seed: int) -> Any: ...


@dataclass(slots=True)
class NativeImageResearchResult:
    shot_id: str
    plan_hash: str
    seed: int
    identity_state: Any
    scene_state: Any
    image: Any


class NativeImageResearchBackend:
    """Execute CINEOS conditioning through an injected native model implementation.

    The repository deliberately does not pretend to contain trained weights yet.
    This class establishes the exact ownership boundary for a future CINEOS model.
    """

    def __init__(self, model: NativeImageModel) -> None:
        self.model = model

    def render(self, plan: NativeImageConditioningPlan) -> NativeImageResearchResult:
        if not isinstance(plan, NativeImageConditioningPlan):
            raise TypeError("plan must be a NativeImageConditioningPlan")
        if not plan.identity_tokens:
            raise ValueError("native image generation requires identity conditioning")
        if not plan.content_hash:
            plan.refresh_hash()

        identity_state = self.model.encode_identity(plan.identity_tokens)
        scene_state = self.model.encode_scene(plan)
        image = self.model.generate(
            identity_state=identity_state,
            scene_state=scene_state,
            seed=plan.seed,
        )
        return NativeImageResearchResult(
            shot_id=plan.shot_id,
            plan_hash=plan.content_hash,
            seed=plan.seed,
            identity_state=identity_state,
            scene_state=scene_state,
            image=image,
        )
