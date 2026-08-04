"""CogVideoX-2B prompt construction."""

from __future__ import annotations

from .performance import PerformanceBrief


def build_prompt(brief: PerformanceBrief, *, max_chars: int = 1800) -> tuple[str, str]:
    s = brief.sections
    subject, action, perf, camera = (
        s["SUBJECT"],
        s["ACTION"],
        s["PERFORMANCE"],
        s["CAMERA"],
    )
    # Order is deliberate: visible subject/action, performance, camera, world.
    parts = [
        f"{camera['size']} shot of {subject['character_id']} "
        f"wearing {subject['wardrobe']}.",
        f"First, {action['primary']}; visible behavior: {action['visible_behavior']}.",
        f"Expression visibly {perf['expression']}; posture {perf['posture']}; "
        f"gesture {perf['gesture']}; looks {perf['eye_line']}.",
    ]
    dialogue = s["DIALOGUE"]
    if dialogue["text"]:
        parts.append(
            "The character visibly speaks one short line with "
            f"{dialogue['delivery']} delivery; visual speaking performance only."
        )
    parts += [
        f"Camera: {camera['angle']}, {camera['lens_intent']}, {camera['movement']}.",
        f"Environment: {s['ENVIRONMENT']['state']}. Lighting: {s['LIGHTING']}.",
        f"Maintain continuity: {s['CONTINUITY']}.",
    ]
    return " ".join(parts)[:max_chars], ", ".join(s["NEGATIVE CONSTRAINTS"])
