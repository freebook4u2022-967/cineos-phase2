"""Deterministic MovieProject/NOVA-to-audio planning."""

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from .dialogue import DialogueCue
from .project import AudioProject


def plan_audio(
    project: Any,
    film_package_id: str,
    *,
    language: str = "en",
    existing: AudioProject | None = None,
) -> AudioProject:
    """Build deterministic cues; preserve matching manually edited dialogue."""
    result = AudioProject(film_package_id, language=language)
    previous = {
        (cue.scene_id, cue.shot_id): cue
        for cue in (existing.dialogue_tracks if existing else [])
    }
    clock = 0.0
    for scene in project.scenes:
        for shot in scene.shots:
            prior = previous.get((scene.scene_id, shot.shot_id))
            if prior is not None:
                result.dialogue_tracks.append(prior)
            elif shot.dialogue:
                # Core projects currently store unstructured dialogue; never invent a
                # speaker. Character identity must be explicit in the scene.
                character = scene.characters[0] if len(scene.characters) == 1 else ""
                cue_seed = (
                    f"{film_package_id}:{scene.scene_id}:{shot.shot_id}:{shot.dialogue}"
                )
                result.dialogue_tracks.append(
                    DialogueCue(
                        scene.scene_id,
                        shot.shot_id,
                        character,
                        shot.dialogue,
                        language,
                        clock,
                        shot.duration,
                        cue_id=str(uuid5(NAMESPACE_URL, cue_seed)),
                        subtitle_text=shot.dialogue,
                    )
                )
            clock += shot.duration
    result.subtitle_metadata = [
        {
            "cue_id": cue.cue_id,
            "text": cue.subtitle_text or cue.line_text,
            "start": cue.start_time,
            "end": cue.end_time,
            "language": cue.language,
        }
        for cue in result.dialogue_tracks
    ]
    return result
