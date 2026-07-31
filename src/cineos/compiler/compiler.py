"""Compile the CINEOS core project model into a Film Package."""

from copy import deepcopy
from typing import Any

from cineos.core import MovieProject, ProjectValidator

from .hashing import build_hashes
from .manifest import FILM_PACKAGE_VERSION, FilmPackage


def _asset(asset: Any, kind: str) -> dict[str, Any]:
    return {
        "asset_id": asset.asset_id,
        "name": asset.name,
        "description": asset.description,
        "metadata": deepcopy(asset.metadata),
        "type": kind,
    }


def compile(project: MovieProject) -> FilmPackage:
    """Validate and deterministically compile *project* without rendering it."""

    if not isinstance(project, MovieProject):
        raise TypeError("project must be a MovieProject")
    ProjectValidator().raise_for_errors(project)

    characters = [_asset(asset, "character") for asset in project.characters]
    locations = [_asset(asset, "location") for asset in project.locations]
    props = [_asset(asset, "prop") for asset in project.props]
    canonical_assets = [
        {
            "asset_id": str(asset.asset_id),
            "type": asset.kind,
            "name": asset.name,
            "version": asset.version,
            "content_hash": asset.content_hash,
        }
        for asset in project.asset_registry.list()
        if not project.asset_ids or asset.asset_id in project.asset_ids
    ]
    scenes = [
        {
            "scene_id": scene.scene_id,
            "title": scene.title,
            "description": scene.description,
            "location": scene.location,
            "characters": list(scene.characters),
            "duration": scene.duration,
            "shots": [shot.shot_id for shot in scene.shots],
        }
        for scene in project.scenes
    ]
    shots = [
        {
            "shot_id": shot.shot_id,
            "scene_id": scene.scene_id,
            "camera": shot.camera,
            "lens": shot.lens,
            "movement": shot.movement,
            "lighting": shot.lighting,
            "action": shot.action,
            "dialogue": shot.dialogue,
            "duration": shot.duration,
            "references": list(shot.references),
        }
        for scene in project.scenes
        for shot in scene.shots
    ]
    payload = {
        "version": FILM_PACKAGE_VERSION,
        "project_metadata": {
            "title": project.title,
            "author": project.author,
            "project_version": project.version,
            "fps": project.fps,
            "resolution": list(project.resolution),
            "aspect_ratio": project.aspect_ratio,
        },
        "scene_manifest": scenes,
        "shot_manifest": shots,
        "character_manifest": characters,
        "location_manifest": locations,
        # Only stable identity metadata enters a Film Package. Reference media paths
        # and image bytes remain in the external asset registry.
        "asset_manifest": [*characters, *locations, *props, *canonical_assets],
        "timeline_manifest": {
            "scene_order": list(project.timeline.scene_order),
            "shot_order": {
                scene_id: list(project.timeline.shot_order[scene_id])
                for scene_id in project.timeline.scene_order
            },
        },
    }
    return FilmPackage(**payload, content_hashes=build_hashes(payload))


class FilmCompiler:
    """Object-oriented façade for applications that inject a compiler."""

    def compile(self, project: MovieProject) -> FilmPackage:
        return compile(project)
