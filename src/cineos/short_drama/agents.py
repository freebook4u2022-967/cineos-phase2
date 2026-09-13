"""Provider-neutral short-drama planning agents."""

from __future__ import annotations

from .models import DramaBrief


class StoryArchitect:
    """Backward-compatible Sprint 1 story shell."""

    def run(self, brief: DramaBrief) -> dict:
        return {
            "premise": brief.premise.strip(),
            "genre": brief.genre,
            "tone": brief.tone,
            "target_duration_seconds": brief.duration_seconds,
            "structure": ["hook", "escalation", "reversal", "climax", "resolution"],
        }


class ScreenwriterAgent:
    """Translate story structure into scenes and dramatic beats."""

    def run(self, story: dict) -> dict:
        beats = []
        scenes = []
        structure = story["structure"]
        for index, beat in enumerate(structure, start=1):
            purpose = self._purpose(beat, story)
            beats.append({"beat": beat, "index": index, "purpose": purpose})
            scenes.append(
                {
                    "scene_id": f"scene-{index:03d}",
                    "beat": beat,
                    "purpose": purpose,
                    "location": (
                        "primary story location" if index == 1 else "previous location"
                    ),
                    "time_of_day": "continuous",
                    "weather": "continuous",
                    "dialogue_intent": self._dialogue_intent(beat),
                    "state_changes": {},
                }
            )
        return {"logline": story["premise"], "beats": beats, "scenes": scenes}

    @staticmethod
    def _purpose(beat: str, story: dict) -> str:
        mapping = {
            "hook": story.get("hook", "create immediate curiosity"),
            "escalation": story.get("stakes", "increase pressure"),
            "reversal": story.get("twist", "change the meaning of prior events"),
            "climax": story.get("climax", "force an irreversible choice"),
            "resolution": story.get("resolution", "show the consequence"),
        }
        return mapping.get(beat, "advance the dramatic question")

    @staticmethod
    def _dialogue_intent(beat: str) -> str:
        return {
            "hook": "withhold exposition; create a question",
            "escalation": "reveal only what increases pressure",
            "reversal": "use silence before explanation",
            "climax": "make dialogue action-oriented and irreversible",
            "resolution": "prefer a visual answer over explanatory dialogue",
        }.get(beat, "serve the immediate objective")


class DirectorAgent:
    """Legacy compatible director shell; Sprint 2 uses DirectorDecisionEngine."""

    def run(self, screenplay: dict, tone: str) -> dict:
        return {
            "tone": tone,
            "performance_rule": "motivated, restrained, continuity-aware",
            "visual_rule": "story-first cinematic coverage",
            "beat_count": len(screenplay["beats"]),
        }


class ShotPlanner:
    def run(
        self,
        screenplay: dict,
        duration_seconds: int,
        direction: dict | None = None,
    ) -> list[dict]:
        beats = screenplay["beats"]
        per_beat = duration_seconds / len(beats)
        decisions = {
            item["beat"]: item for item in (direction or {}).get("decisions", [])
        }
        shots = []
        for i, item in enumerate(beats, start=1):
            decision = decisions.get(item["beat"], {})
            shots.append(
                {
                    "shot_id": f"shot-{i:03d}",
                    "scene_id": f"scene-{i:03d}",
                    "beat": item["beat"],
                    "purpose": item.get("purpose"),
                    "duration_seconds": per_beat,
                    "shot_size": decision.get("shot_size", "medium"),
                    "lens": decision.get("lens", "50mm"),
                    "camera_movement": decision.get("camera_movement", "static"),
                    "performance": decision.get(
                        "performance", "play the objective truthfully"
                    ),
                    "lighting_intent": decision.get(
                        "lighting_intent", "motivated naturalistic light"
                    ),
                    "status": "planned",
                }
            )
        return shots


class ContinuitySupervisor:
    def run(self, shots: list[dict], scene_states: list | None = None) -> dict:
        issues = []
        if len({shot["shot_id"] for shot in shots}) != len(shots):
            issues.append("duplicate shot identifiers")
        return {
            "status": "pass" if not issues else "fail",
            "issues": issues,
            "shot_order": [shot["shot_id"] for shot in shots],
            "scene_state_count": len(scene_states or []),
            "rules": [
                "preserve approved character identity",
                "preserve wardrobe and props unless scripted",
                "preserve spatial and temporal causality",
                "preserve character knowledge unless a scene explicitly changes it",
                "preserve physical state until an explicit scripted change",
            ],
        }
