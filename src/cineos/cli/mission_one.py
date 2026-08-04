from __future__ import annotations

from pathlib import Path

from cineos.mission_one.brief import DirectedSceneBrief
from cineos.mission_one.compiler import compile_scene
from cineos.renderers.colab import (
    ColabRenderConfig,
    assemble,
    export_package,
    verify_results,
)
from cineos.renderers.colab.package import ColabRenderPackage
from cineos.renderers.colab.serializer import dump_json, load_json


def run(
    command: str,
    *,
    source: Path,
    output: Path | None = None,
    output_dir: Path | None = None,
    dry_run: bool = False,
    model: str = "THUDM/CogVideoX-2b",
    resolution: str = "720x480",
    fps: int = 8,
    steps: int = 50,
    seed: int = 42,
) -> dict:
    if command == "compile":
        brief = DirectedSceneBrief.from_dict(load_json(source))
        config = ColabRenderConfig(model, fps, resolution, steps, 6.0, seed)
        package = compile_scene(brief, config)
        destination = (output_dir or Path(".")) / "package.json"
        if not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            dump_json(package.to_dict(), destination)
        return {
            "package": str(destination),
            "shot_count": len(package.shots),
            "dry_run": dry_run,
        }
    if command == "inspect":
        package = ColabRenderPackage.from_dict(load_json(source))
        return {
            "project_id": package.project_id,
            "scene_id": package.scene_id,
            "shot_count": len(package.shots),
            "duration": sum(x.duration for x in package.shots),
            "model_id": package.config.model_id,
        }
    if command == "export-colab":
        if output is None:
            raise ValueError("--output is required")
        if not dry_run:
            export_package(ColabRenderPackage.from_dict(load_json(source)), output)
        return {"output": str(output), "dry_run": dry_run}
    if command == "verify":
        return verify_results(source)
    if command == "assemble":
        if output is None:
            raise ValueError("--output is required")
        if not dry_run:
            assemble(source, output, fps)
        return {"output": str(output), "dry_run": dry_run}
    raise ValueError(f"unknown Mission One command: {command}")
