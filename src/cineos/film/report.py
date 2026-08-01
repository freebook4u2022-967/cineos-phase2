"""Deterministic human- and machine-readable build reporting."""

from typing import Any

from .build import FilmBuild


def build_report(build: FilmBuild) -> dict[str, Any]:
    return {
        "build_id": build.build_id,
        "project_id": build.project_id,
        "renderer_id": build.renderer_id,
        "status": str(build.status),
        "content_hash": build.content_hash,
        "shots": [
            {
                "shot_id": shot.shot_id,
                "attempts": shot.attempt_count,
                "render_status": shot.render_status,
                "validation_status": shot.validation_status,
                "selected_output": shot.selected_output,
            }
            for shot in build.shot_states
        ],
        "warnings": list(build.warnings),
        "failures": list(build.failures),
        "outputs": dict(sorted(build.output_files.items())),
    }
