"""Thin command adapters that compose existing CINEOS APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cineos import __version__
from cineos.atlas import (
    AtlasRuntime,
    BaseRenderer,
    Range,
    RendererAdapter,
    RendererCapabilities,
    RendererRegistry,
    Resolution,
)
from cineos.compiler import compile as compile_project
from cineos.compiler import load as load_package
from cineos.compiler import save
from cineos.core import (
    Character,
    Environment,
    MovieProject,
    ProjectValidationError,
    ProjectValidator,
    Prop,
    Scene,
    Shot,
    Timeline,
)

from .errors import CLIError, ExitCode
from .output import Output


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CLIError(
            f"file not found: {path}",
            code=ExitCode.INPUT,
            hint="Check the path and try again.",
        ) from error
    except (OSError, UnicodeError) as error:
        raise CLIError(f"cannot read {path}: {error}", code=ExitCode.INPUT) from error
    except json.JSONDecodeError as error:
        raise CLIError(
            f"invalid JSON in {path} at line {error.lineno}, "
            f"column {error.colno}: {error.msg}",
            code=ExitCode.INPUT,
            hint="Correct the JSON syntax and retry.",
        ) from error
    if not isinstance(value, dict):
        raise CLIError(
            "project JSON must contain an object at its root", code=ExitCode.INPUT
        )
    return value


def load_project(path: Path) -> MovieProject:
    """Deserialize the documented JSON representation into the core model."""

    value = _read_json(path)
    try:
        scenes = [
            Scene(
                scene_id=scene["scene_id"],
                title=scene["title"],
                description=scene.get("description", ""),
                location=scene.get("location"),
                characters=list(scene.get("characters", [])),
                duration=scene.get("duration", 0.0),
                shots=[Shot(**shot) for shot in scene.get("shots", [])],
            )
            for scene in value.get("scenes", [])
        ]
        timeline_value = value.get("timeline")
        timeline = (
            Timeline(
                scene_order=list(timeline_value.get("scene_order", [])),
                shot_order={
                    key: list(items)
                    for key, items in timeline_value.get("shot_order", {}).items()
                },
            )
            if timeline_value is not None
            else Timeline(
                scene_order=[scene.scene_id for scene in scenes],
                shot_order={
                    scene.scene_id: [shot.shot_id for shot in scene.shots]
                    for scene in scenes
                },
            )
        )
        resolution = value.get("resolution", [1920, 1080])
        return MovieProject(
            title=value["title"],
            author=value["author"],
            version=value.get("version", "1.0"),
            fps=value.get("fps", 24.0),
            resolution=tuple(resolution),
            aspect_ratio=value.get("aspect_ratio", "16:9"),
            characters=[Character(**item) for item in value.get("characters", [])],
            locations=[Environment(**item) for item in value.get("locations", [])],
            props=[Prop(**item) for item in value.get("props", [])],
            scenes=scenes,
            timeline=timeline,
        )
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise CLIError(
            f"invalid project structure: {error}",
            code=ExitCode.INPUT,
            hint="See the project JSON example in README.md.",
        ) from error


def validate(path: Path, output: Output) -> None:
    project = load_project(path)
    errors = ProjectValidator().validate(project)
    if errors:
        raise CLIError(
            "project validation failed: " + "; ".join(errors),
            code=ExitCode.VALIDATION,
            hint="Fix the reported model errors before compiling.",
        )
    output.success(f"Project is valid: {path}", project=str(path))


def compile(path: Path, destination: Path, output: Output) -> None:
    project = load_project(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        package = compile_project(project)
    except ProjectValidationError as error:
        raise CLIError(
            f"project validation failed: {error}",
            code=ExitCode.VALIDATION,
            hint="Run 'cineos validate' and fix the reported model errors.",
        ) from error
    save(package, destination)
    output.success(
        f"Film Package written to {destination}",
        output=str(destination),
        package_hash=package.content_hashes["package"],
    )


def render(package_path: Path, output_dir: Path, output: Output) -> list[Path]:
    try:
        package = load_package(package_path)
    except (OSError, ValueError, TypeError) as error:
        raise CLIError(
            f"cannot load Film Package {package_path}: {error}",
            code=ExitCode.VALIDATION,
            hint="Compile or repair the package before rendering.",
        ) from error
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = RendererRegistry()
    registry.register("preview", lambda: _PreviewRenderer(output_dir))
    adapter = RendererAdapter(registry.create("preview"))
    adapter.initialize()
    adapter.load_model()
    adapter.warmup()
    try:
        job = AtlasRuntime().execute(
            package, adapter.render, job_id="cineos-cli-preview"
        )
    finally:
        adapter.shutdown()
    paths = [Path(job.results[shot_id]) for shot_id in job.completed]
    manifest = {
        "format": "cineos-preview-render-v1",
        "files": [path.name for path in paths],
    }
    (output_dir / "render-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    output.success(
        f"Rendered {len(paths)} preview shot(s) to {output_dir}",
        output_dir=str(output_dir),
        rendered=len(paths),
    )
    return paths


def assemble(render_dir: Path, destination: Path, output: Output) -> None:
    manifest_path = render_dir / "render-manifest.json"
    manifest = _read_json(manifest_path)
    files = manifest.get("files")
    if manifest.get("format") != "cineos-preview-render-v1" or not isinstance(
        files, list
    ):
        raise CLIError(
            f"invalid render manifest: {manifest_path}", code=ExitCode.VALIDATION
        )
    try:
        frames = [(render_dir / name).read_bytes() for name in files]
    except (OSError, TypeError) as error:
        raise CLIError(
            f"cannot read rendered output: {error}", code=ExitCode.INPUT
        ) from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"CINEOS-PREVIEW-MOVIE\n" + b"".join(frames))
    output.success(
        f"Preview movie assembled at {destination}",
        output=str(destination),
        shots=len(frames),
    )


def demo(output_dir: Path, output: Output) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shot = Shot("demo-shot", action="CINEOS preview title card", duration=1.0)
    scene = Scene("demo-scene", "CINEOS Demo", shots=[shot], duration=1.0)
    project = MovieProject(
        "CINEOS Demo",
        "CINEOS",
        scenes=[scene],
        timeline=Timeline([scene.scene_id], {scene.scene_id: [shot.shot_id]}),
    )
    package_path = output_dir / "film-package.json"
    save(compile_project(project), package_path)
    render_dir = output_dir / "renders"
    quiet = _QuietOutput(json_mode=output.json_mode, stderr=output.stderr)
    render(package_path, render_dir, quiet)
    movie_path = output_dir / "demo.mp4"
    assemble(
        render_dir,
        movie_path,
        quiet,
    )
    output.success(
        f"Demo pipeline completed in {output_dir}",
        output_dir=str(output_dir),
        package=str(package_path),
        movie=str(movie_path),
    )


def version(output: Output) -> None:
    output.success(f"cineos {__version__}", version=__version__)


class _QuietOutput(Output):
    """Suppress successful intermediate stages of a composite command."""

    def success(self, message: str, **details: Any) -> None:
        pass


class _PreviewRenderer(BaseRenderer):
    """Deterministic, CPU-only renderer used for pipeline inspection."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    @property
    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities(
            supported_resolution=(Resolution(1920, 1080),),
            supported_duration=Range(0, float("inf")),
            supported_fps=(24.0,),
            supported_features=frozenset({"metadata-preview"}),
        )

    def initialize(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, model: str | None = None, **options: Any) -> None:
        """The preview backend intentionally has no model to load."""

    def warmup(self) -> None:
        """The CPU-only preview backend needs no warmup."""

    def render(self, request: Any) -> str:
        destination = self.output_dir / f"{request.shot_id}.preview.json"
        payload = {
            "renderer": "preview",
            "scene_id": request.scene_id,
            "shot_id": request.shot_id,
            "shot": dict(request.shot),
        }
        destination.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        return str(destination)

    def shutdown(self) -> None:
        """The preview backend owns no persistent resources."""
