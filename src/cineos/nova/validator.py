"""Cross-plan production validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cineos.atlas import RendererCapabilities

from .exceptions import PlanValidationError

if TYPE_CHECKING:
    from .director import DirectorPlan


class NOVAValidator:
    def __init__(self, capabilities: RendererCapabilities | None = None) -> None:
        self.capabilities = capabilities

    def errors(self, plan: DirectorPlan) -> list[str]:
        errors: list[str] = []
        duration = sum(shot.duration for shot in plan.shots)
        tolerance = max(0.01, plan.brief.target_duration * 0.01)
        if abs(duration - plan.brief.target_duration) > tolerance:
            errors.append(
                f"duration {duration:.3f}s differs from target "
                f"{plan.brief.target_duration:.3f}s"
            )
        scene_ids = {scene.scene_id for scene in plan.scenes}
        for shot in plan.shots:
            if shot.scene_id not in scene_ids:
                errors.append(f"shot {shot.shot_id} references an unknown scene")
            if self.capabilities:
                unsupported = (
                    shot.renderer_capability_requirements
                    - self.capabilities.supported_features
                )
                if unsupported:
                    errors.append(
                        f"shot {shot.shot_id} requests unsupported renderer features: "
                        + ", ".join(sorted(unsupported))
                    )
        previous = None
        for scene in plan.scenes:
            if previous and scene.continuity_inputs != previous:
                errors.append(f"scene {scene.scene_id} has unresolved continuity input")
            previous = scene.continuity_outputs
        return errors

    def validate(self, plan: DirectorPlan) -> None:
        if errors := self.errors(plan):
            raise PlanValidationError("; ".join(errors))
