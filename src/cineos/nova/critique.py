"""Structured, explainable NOVA critique pass."""

from dataclasses import dataclass, field

from cineos.atlas import RendererCapabilities

from .director import DirectorPlan


@dataclass(slots=True)
class CritiqueFinding:
    code: str
    severity: str
    message: str
    scene_id: str | None = None
    shot_id: str | None = None
    evidence: dict[str, object] = field(default_factory=dict)
    suggested_action: str = "review"


class NOVACritic:
    def critique(
        self,
        plan: DirectorPlan,
        capabilities: RendererCapabilities | None = None,
    ) -> list[CritiqueFinding]:
        findings: list[CritiqueFinding] = []
        for previous, current in zip(plan.shots, plan.shots[1:], strict=False):
            if (previous.framing, previous.camera_movement) == (
                current.framing,
                current.camera_movement,
            ):
                findings.append(
                    CritiqueFinding(
                        "repetitive-shot-language",
                        "warning",
                        "Adjacent shots repeat framing and movement.",
                        current.scene_id,
                        current.shot_id,
                        suggested_action="vary-camera",
                    )
                )
        durations = [scene.estimated_duration for scene in plan.scenes]
        if durations and max(durations) > min(durations) * 2:
            findings.append(
                CritiqueFinding(
                    "pacing-imbalance",
                    "warning",
                    "Scene durations are materially imbalanced.",
                    suggested_action="rebalance-duration",
                )
            )
        escalation = [scene.pacing.escalation for scene in plan.scenes]
        if any(
            right <= left
            for left, right in zip(escalation, escalation[1:], strict=False)
        ):
            findings.append(
                CritiqueFinding(
                    "missing-escalation",
                    "error",
                    "Dramatic escalation does not increase.",
                    suggested_action="increase-escalation",
                )
            )
        for previous, current in zip(plan.scenes, plan.scenes[1:], strict=False):
            if current.continuity_inputs != previous.continuity_outputs:
                findings.append(
                    CritiqueFinding(
                        "unresolved-continuity",
                        "error",
                        "Scene input does not carry the previous output.",
                        current.scene_id,
                        suggested_action="repair-continuity",
                    )
                )
        seen: dict[tuple[str, str], str] = {}
        for scene in plan.scenes:
            key = (scene.narrative_purpose, scene.dramatic_beat)
            if key in seen:
                findings.append(
                    CritiqueFinding(
                        "scene-redundancy",
                        "warning",
                        "Scene duplicates an existing purpose and beat.",
                        scene.scene_id,
                        suggested_action="remove-redundancy",
                    )
                )
            seen[key] = scene.scene_id
            if not scene.participating_character_ids or not scene.narrative_purpose:
                findings.append(
                    CritiqueFinding(
                        "missing-character-motivation",
                        "error",
                        "Scene lacks a motivated participating character.",
                        scene.scene_id,
                        suggested_action="clarify-motivation",
                    )
                )
        duration = sum(item.duration for item in plan.shots)
        if duration > plan.brief.target_duration * 1.01:
            findings.append(
                CritiqueFinding(
                    "excessive-duration",
                    "error",
                    "Plan exceeds its target duration.",
                    evidence={"duration": duration},
                    suggested_action="trim-duration",
                )
            )
        if capabilities:
            for shot in plan.shots:
                unsupported = (
                    shot.renderer_capability_requirements
                    - capabilities.supported_features
                )
                if unsupported:
                    findings.append(
                        CritiqueFinding(
                            "unsupported-renderer-request",
                            "error",
                            "Shot requests unsupported capabilities.",
                            shot.scene_id,
                            shot.shot_id,
                            {"features": sorted(unsupported)},
                            "remove-unsupported-request",
                        )
                    )
        return findings
