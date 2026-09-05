"""Small command adapter for creating renderer-independent drama packages."""

from __future__ import annotations

import json
from pathlib import Path

from .models import DramaBrief
from .orchestrator import ShortDramaOrchestrator


def create_drama_plan(
    premise: str,
    *,
    duration_seconds: int = 180,
    genre: str = "drama",
    tone: str = "cinematic",
) -> dict:
    """Create a JSON-safe drama plan from one creative premise."""
    brief = DramaBrief(
        premise=premise,
        duration_seconds=duration_seconds,
        genre=genre,
        tone=tone,
    )
    return ShortDramaOrchestrator().plan(brief).to_dict()


def write_drama_plan(
    premise: str,
    output: Path,
    *,
    duration_seconds: int = 180,
    genre: str = "drama",
    tone: str = "cinematic",
) -> Path:
    """Persist a canonical human-readable drama package for later CINEOS stages."""
    payload = create_drama_plan(
        premise,
        duration_seconds=duration_seconds,
        genre=genre,
        tone=tone,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
