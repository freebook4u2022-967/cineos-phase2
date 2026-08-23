"""Director decision engine for short-drama coverage."""

from __future__ import annotations


class DirectorDecisionEngine:
    """Turn dramatic beats into explicit story-first camera and performance choices."""

    SHOT_LANGUAGE = {
        "hook": ("wide", "35mm", "slow push-in"),
        "escalation": ("medium", "50mm", "controlled handheld drift"),
        "reversal": ("close-up", "85mm", "locked-off hold"),
        "climax": ("medium close-up", "50mm", "decisive push-in"),
        "resolution": ("wide", "35mm", "slow pull-back"),
    }

    def run(self, screenplay: dict, tone: str) -> dict:
        decisions = []
        for item in screenplay["beats"]:
            beat = item["beat"]
            shot_size, lens, movement = self.SHOT_LANGUAGE.get(
                beat, ("medium", "50mm", "static")
            )
            decisions.append(
                {
                    "beat": beat,
                    "shot_size": shot_size,
                    "lens": lens,
                    "camera_movement": movement,
                    "performance": self._performance_for(beat),
                    "lighting_intent": self._lighting_for(beat, tone),
                    "blocking_rule": "movement must be motivated by a change in objective",
                }
            )
        return {
            "tone": tone,
            "visual_rule": "story-first cinematic coverage",
            "performance_rule": "restrained, motivated, continuity-aware",
            "decisions": decisions,
        }

    @staticmethod
    def _performance_for(beat: str) -> str:
        return {
            "hook": "play recognition before explanation",
            "escalation": "increase urgency without announcing it",
            "reversal": "let the discovery land before dialogue",
            "climax": "commit physically to the irreversible choice",
            "resolution": "reduce movement and let consequence register",
        }.get(beat, "play the immediate objective truthfully")

    @staticmethod
    def _lighting_for(beat: str, tone: str) -> str:
        if beat in {"reversal", "climax"}:
            return f"heighten contrast while preserving motivated sources; tone={tone}"
        return f"naturalistic motivated light; tone={tone}"
