"""Targeted revision without wholesale regeneration."""

from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime

from .critique import CritiqueFinding
from .director import DirectorPlan


class NOVARevisionEngine:
    def revise(
        self, plan: DirectorPlan, findings: list[CritiqueFinding]
    ) -> DirectorPlan:
        revised = deepcopy(plan)
        selected = {(item.scene_id, item.shot_id): item for item in findings}
        for (scene_id, shot_id), finding in selected.items():
            if shot_id:
                shot = next(item for item in revised.shots if item.shot_id == shot_id)
                if finding.suggested_action == "vary-camera":
                    shot.framing = (
                        "close-up" if shot.framing != "close-up" else "wide shot"
                    )
                    shot.rationale = (
                        "Camera language varied in response to an accepted critique."
                    )
                elif finding.suggested_action == "remove-unsupported-request":
                    shot.renderer_capability_requirements.clear()
                elif finding.suggested_action == "trim-duration":
                    shot.duration *= 0.9
            if scene_id:
                scene = next(
                    item for item in revised.scenes if item.scene_id == scene_id
                )
                if finding.suggested_action == "repair-continuity":
                    index = revised.scenes.index(scene)
                    if index:
                        scene.continuity_inputs = deepcopy(
                            revised.scenes[index - 1].continuity_outputs
                        )
                elif finding.suggested_action == "increase-escalation":
                    scene.pacing.escalation = min(1.0, scene.pacing.escalation + 0.2)
        revised.revision_history.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "findings": [asdict(item) for item in findings],
                "scope": sorted(
                    {item.scene_id or item.shot_id or "plan" for item in findings}
                ),
            }
        )
        by_id = {item.shot_id: item for item in revised.shots}
        for scene in revised.project.scenes:
            for core_shot in scene.shots:
                shot = by_id[core_shot.shot_id]
                core_shot.camera = shot.framing
                core_shot.lens = shot.lens
                core_shot.movement = shot.camera_movement
                core_shot.lighting = shot.lighting_intent
                core_shot.action = shot.action
                core_shot.dialogue = shot.dialogue_intent
                core_shot.duration = shot.duration
            scene.duration = scene.shot_duration
        return revised
