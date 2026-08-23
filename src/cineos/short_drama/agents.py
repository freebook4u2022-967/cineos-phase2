"""Deterministic Sprint 1 agents.

These contracts intentionally avoid model/provider dependencies. Later adapters may
replace each implementation while preserving the orchestration boundary.
"""

from .models import DramaBrief


class StoryArchitect:
    def run(self, brief: DramaBrief) -> dict:
        return {
            "premise": brief.premise.strip(),
            "genre": brief.genre,
            "tone": brief.tone,
            "target_duration_seconds": brief.duration_seconds,
            "structure": ["hook", "escalation", "reversal", "climax", "resolution"],
        }


class ScreenwriterAgent:
    def run(self, story: dict) -> dict:
        beats = story["structure"]
        return {
            "logline": story["premise"],
            "beats": [{"beat": beat, "index": i + 1} for i, beat in enumerate(beats)],
        }


class DirectorAgent:
    def run(self, screenplay: dict, tone: str) -> dict:
        return {
            "tone": tone,
            "performance_rule": "motivated, restrained, continuity-aware",
            "visual_rule": "story-first cinematic coverage",
            "beat_count": len(screenplay["beats"]),
        }


class ShotPlanner:
    def run(self, screenplay: dict, duration_seconds: int) -> list[dict]:
        beats = screenplay["beats"]
        per_beat = duration_seconds / len(beats)
        return [
            {
                "shot_id": f"shot-{i:03d}",
                "beat": item["beat"],
                "duration_seconds": per_beat,
                "status": "planned",
            }
            for i, item in enumerate(beats, start=1)
        ]


class ContinuitySupervisor:
    def run(self, shots: list[dict]) -> dict:
        return {
            "status": "pass",
            "shot_order": [shot["shot_id"] for shot in shots],
            "rules": [
                "preserve approved character identity",
                "preserve wardrobe and props unless scripted",
                "preserve spatial and temporal causality",
            ],
        }
