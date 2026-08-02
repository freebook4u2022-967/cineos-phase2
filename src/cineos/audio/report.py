"""Audio production reporting."""

from dataclasses import asdict
from typing import Any

from .project import AudioProject
from .validator import ValidationReport


def production_report(
    project: AudioProject, validation: ValidationReport | None = None
) -> dict[str, Any]:
    return {
        "project_id": project.project_id,
        "film_package_id": project.film_package_id,
        "content_hash": project.content_hash,
        "language": project.language,
        "sample_rate": project.sample_rate,
        "channel_layout": project.channel_layout,
        "cue_counts": {
            "dialogue": len(project.dialogue_tracks),
            "ambience": len(project.ambience_tracks),
            "effects": len(project.effects_tracks),
            "music": len(project.music_tracks),
        },
        "validation": asdict(validation) if validation else None,
        "visual_lip_sync_rendered": False,
    }
