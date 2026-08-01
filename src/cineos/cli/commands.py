"""Thin command adapters that compose existing CINEOS APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cineos import __version__
from cineos.assets import AssetRegistry as ProductionAssetRegistry
from cineos.assets import Character as ProductionCharacter
from cineos.assets import Environment as ProductionEnvironment
from cineos.assets import ReferenceImage as ProductionReference
from cineos.assets.storage import asset_to_dict
from cineos.assets.storage import load as load_assets
from cineos.assets.storage import save as save_assets
from cineos.atlas import (
    AtlasRuntime,
    BaseRenderer,
    Range,
    RendererAdapter,
    RendererCapabilities,
    RendererRegistry,
    Resolution,
)
from cineos.cinedna import CineDNABuilder, CineDNARegistry
from cineos.cinedna.serializer import profile_to_dict
from cineos.cinedna.serializer import save as save_cinedna
from cineos.compiler import compile as compile_project
from cineos.compiler import load as load_package
from cineos.compiler import save
from cineos.conditioning import ConditioningBuilder, ConditioningValidator
from cineos.conditioning.serializer import load as load_conditioning
from cineos.conditioning.serializer import package_to_dict as conditioning_to_dict
from cineos.conditioning.serializer import save as save_conditioning
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
from cineos.hardware import probe as probe_hardware
from cineos.hardware import to_json as hardware_to_json
from cineos.hardware import to_text as hardware_to_text
from cineos.plugins import Plugin, PluginContext, PluginManager, PluginMetadata

from .errors import CLIError, ExitCode
from .output import Output


def condition(
    command: str,
    output: Output,
    *,
    package_path: Path,
    registry_path: Path,
    profiles_path: Path,
    shot_id: str | None = None,
    conditioning_path: Path | None = None,
    destination: Path | None = None,
) -> None:
    """Build, export, validate, or display reference conditioning."""
    if command in {"validate", "show"}:
        if conditioning_path is None:
            raise CLIError("conditioning package path is required", code=ExitCode.USAGE)
        package = load_conditioning(conditioning_path)
        ConditioningValidator().raise_for_errors(package)
        if command == "show":
            output.success(
                json.dumps(conditioning_to_dict(package), sort_keys=True, indent=2),
                package=conditioning_to_dict(package),
            )
        else:
            output.success(
                f"Conditioning package is valid: {conditioning_path}",
                package=str(conditioning_path),
            )
        return
    if shot_id is None:
        raise CLIError("shot ID is required", code=ExitCode.USAGE)
    package = ConditioningBuilder(
        load_package(package_path),
        load_assets(registry_path),
        CineDNARegistry.load(profiles_path),
    ).build(shot_id)
    target = destination or Path(f"{shot_id}.conditioning.json")
    save_conditioning(package, target)
    output.success(
        f"Conditioning package written to {target}",
        output=str(target),
        content_hash=package.content_hash,
    )


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
        registry_value = value.get("asset_registry")
        if registry_value is not None and not isinstance(registry_value, str):
            raise TypeError("asset_registry must be a file path")
        registry_path = (
            (path.parent / registry_value).resolve() if registry_value else None
        )
        production_assets = (
            load_assets(registry_path)
            if registry_path is not None
            else ProductionAssetRegistry()
        )
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
            asset_registry=production_assets,
            asset_ids=list(value.get("asset_ids", [])),
            cinedna_ids=list(value.get("cinedna_ids", [])),
        )
    except (KeyError, TypeError, ValueError, AttributeError, OSError) as error:
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


def render(
    package_path: Path,
    output_dir: Path,
    output: Output,
    *,
    runtime_log: Path | None = None,
) -> list[Path]:
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
    runtime = AtlasRuntime()
    plugin_manager = PluginManager()
    plugin_context = PluginContext(
        services={"renderer_registry": registry, "atlas_runtime": runtime},
        settings={"command": "render", "output_dir": str(output_dir)},
    )
    plugin_manager.register(_PreviewRendererPlugin(output_dir, registry, package))
    plugin_manager.discover()
    plugin_manager.activate_all(plugin_context)
    adapter = RendererAdapter(registry.create("preview"))
    try:
        adapter.initialize()
        adapter.load_model()
        adapter.warmup()
        metadata = package.project_metadata
        adapter.capabilities.negotiate(
            resolution=tuple(metadata["resolution"]),
            duration=max(
                (task.shot["duration"] for task in runtime.prepare(package).tasks),
                default=0,
            ),
            fps=metadata["fps"],
            features=("metadata-preview",),
        )
        job = runtime.execute(package, adapter.render, job_id="cineos-cli-preview")
    finally:
        try:
            adapter.shutdown()
        finally:
            plugin_manager.deactivate_all(plugin_context)
    paths = [Path(job.results[shot_id]) for shot_id in job.completed]
    manifest = {
        "format": "cineos-preview-render-v1",
        "files": [path.name for path in paths],
    }
    (output_dir / "render-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    log_path = runtime_log or output_dir / "runtime-log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "format": "cineos-runtime-log-v1",
                "job_id": job.job_id,
                "state": job.state.value,
                "completed": job.completed,
                "progress": job.progress,
                "results": job.results,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
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
    project_path = output_dir / "project.json"
    project_path.write_text(
        json.dumps(_project_to_dict(project), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    package_path = output_dir / "film-package.json"
    quiet = _QuietOutput(json_mode=output.json_mode, stderr=output.stderr)
    compile(project_path, package_path, quiet)
    render_dir = output_dir / "renders"
    runtime_log = output_dir / "runtime-log.json"
    render(package_path, render_dir, quiet, runtime_log=runtime_log)
    movie_path = output_dir / "demo.mp4"
    assemble(
        render_dir,
        movie_path,
        quiet,
    )
    output.success(
        f"Demo pipeline completed in {output_dir}",
        output_dir=str(output_dir),
        project=str(project_path),
        package=str(package_path),
        runtime_log=str(runtime_log),
        movie=str(movie_path),
    )


def _project_to_dict(project: MovieProject) -> dict[str, Any]:
    """Serialize the core fields accepted by :func:`load_project`."""

    return {
        "title": project.title,
        "author": project.author,
        "version": project.version,
        "fps": project.fps,
        "resolution": list(project.resolution),
        "aspect_ratio": project.aspect_ratio,
        "cinedna_ids": [str(value) for value in project.cinedna_ids],
        "scenes": [
            {
                "scene_id": scene.scene_id,
                "title": scene.title,
                "description": scene.description,
                "location": scene.location,
                "characters": scene.characters,
                "duration": scene.duration,
                "shots": [
                    {
                        "shot_id": shot.shot_id,
                        "camera": shot.camera,
                        "lens": shot.lens,
                        "movement": shot.movement,
                        "lighting": shot.lighting,
                        "action": shot.action,
                        "dialogue": shot.dialogue,
                        "duration": shot.duration,
                        "references": shot.references,
                    }
                    for shot in scene.shots
                ],
            }
            for scene in project.scenes
        ],
        "timeline": {
            "scene_order": project.timeline.scene_order,
            "shot_order": project.timeline.shot_order,
        },
    }


def version(output: Output) -> None:
    output.success(f"cineos {__version__}", version=__version__)


def hardware_report(destination: Path | None, verbose: bool, output: Output) -> None:
    """Probe local capabilities and optionally persist the deterministic report."""

    report = probe_hardware(destination.parent if destination else None)
    serialized = hardware_to_json(report)
    if destination is not None:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(serialized, encoding="utf-8")
        except OSError as error:
            raise CLIError(
                f"cannot write hardware report {destination}: {error}",
                code=ExitCode.EXECUTION,
            ) from error
    if output.json_mode:
        output.stdout.write(serialized)
    else:
        output.stdout.write(hardware_to_text(report, verbose=verbose))
        if destination is not None:
            output.stdout.write(f"JSON report written to {destination}\n")


def assets(
    action: str,
    registry_path: Path,
    output: Output,
    destination: Path | None = None,
    *,
    manifest: Path | None = None,
    asset_id: str | None = None,
) -> None:
    """Create, inspect, validate, or export a persisted asset registry."""

    try:
        registry = (
            load_assets(registry_path)
            if registry_path.exists()
            else ProductionAssetRegistry()
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise CLIError(
            f"cannot load asset registry {registry_path}: {error}",
            code=ExitCode.INPUT,
        ) from error
    if action in {"add-character", "add-environment"}:
        assert manifest is not None
        value = _read_json(manifest)
        cls = (
            ProductionCharacter if action == "add-character" else ProductionEnvironment
        )
        try:
            references = [
                ProductionReference(**item)
                for item in value.get("references", value.get("reference_images", []))
            ]
            asset = cls(
                **{
                    key: value[key]
                    for key in (
                        "asset_id",
                        "name",
                        "description",
                        "version",
                        "tags",
                        "metadata",
                        "created_at",
                        "updated_at",
                        "content_hash",
                    )
                    if key in value
                },
                reference_images=references,
            )
            registry.register(asset)
            save_assets(registry, registry_path)
        except (TypeError, ValueError) as error:
            raise CLIError(
                f"invalid asset manifest: {error}", code=ExitCode.INPUT
            ) from error
        output.success(
            f"Registered {asset.kind} {asset.name} ({asset.asset_id})",
            asset=asset_to_dict(asset),
            registry=str(registry_path),
        )
    elif action == "show":
        assert asset_id is not None
        try:
            item = asset_to_dict(registry.retrieve(asset_id))
        except (KeyError, ValueError) as error:
            raise CLIError(
                f"asset not found: {asset_id}", code=ExitCode.INPUT
            ) from error
        if output.json_mode:
            output.success(f"Asset {asset_id}", asset=item)
        else:
            output.stdout.write(
                f"{item['name']}\nID: {item['asset_id']}\nType: {item['type']}\n"
                f"Version: {item['version']}\nDescription: {item['description']}\n"
            )
    elif action == "validate":
        if errors := registry.validate():
            raise CLIError(
                "asset validation failed: " + "; ".join(errors),
                code=ExitCode.VALIDATION,
            )
        output.success(
            f"Asset registry is valid: {registry_path}", assets=len(registry)
        )
    elif action == "export":
        assert destination is not None
        try:
            save_assets(registry, destination)
        except OSError as error:
            raise CLIError(
                f"cannot export assets: {error}", code=ExitCode.EXECUTION
            ) from error
        output.success(
            f"Exported {len(registry)} asset(s) to {destination}",
            assets=len(registry),
            output=str(destination),
        )
    else:
        items = [
            {"asset_id": str(asset.asset_id), "type": asset.kind, "name": asset.name}
            for asset in registry.list()
        ]
        if output.json_mode:
            output.success(f"Found {len(items)} asset(s)", assets=items)
        else:
            for item in items:
                output.stdout.write(
                    f"{item['asset_id']}\t{item['type']}\t{item['name']}\n"
                )


def cinedna(
    action: str,
    asset_registry_path: Path,
    profile_registry_path: Path,
    output: Output,
    *,
    character_id: str | None = None,
    destination: Path | None = None,
) -> None:
    """Build and inspect persistent character identity profiles."""

    try:
        registry = (
            CineDNARegistry.load(profile_registry_path)
            if profile_registry_path.exists()
            else CineDNARegistry()
        )
        if action == "build":
            if not asset_registry_path.exists():
                raise FileNotFoundError(asset_registry_path)
            assets_registry = load_assets(asset_registry_path)
            asset = assets_registry.retrieve(character_id or "")
            if not isinstance(asset, ProductionCharacter):
                raise TypeError("asset is not a character")
            profile = CineDNABuilder().build(asset)
            if profile.character_uuid in {
                item.character_uuid for item in registry.list()
            }:
                registry.update(profile)
            else:
                registry.register(profile)
            registry.save(profile_registry_path)
            output.success(
                f"Built CineDNA for {profile.display_name} ({profile.character_uuid})",
                profile=profile_to_dict(profile),
                registry=str(profile_registry_path),
            )
            return
        if action == "list":
            items = [
                {
                    "character_uuid": str(item.character_uuid),
                    "display_name": item.display_name,
                    "profile_version": item.profile_version,
                    "content_hash": item.content_hash,
                }
                for item in registry.list()
            ]
            if output.json_mode:
                output.success(f"Found {len(items)} CineDNA profile(s)", profiles=items)
            else:
                for item in items:
                    output.stdout.write(
                        f"{item['character_uuid']}\t{item['profile_version']}\t"
                        f"{item['display_name']}\n"
                    )
            return
        profile = registry.retrieve(character_id or "")
        if action == "validate":
            errors = registry.validate(profile.character_uuid)
            if errors:
                raise CLIError(
                    "CineDNA validation failed: " + "; ".join(errors),
                    code=ExitCode.VALIDATION,
                )
            output.success(
                f"CineDNA profile is valid: {profile.character_uuid}",
                character_uuid=str(profile.character_uuid),
            )
        elif action == "export":
            assert destination is not None
            save_cinedna(profile, destination)
            output.success(
                f"Exported CineDNA profile to {destination}", output=str(destination)
            )
        else:
            item = profile_to_dict(profile)
            if output.json_mode:
                output.success(f"CineDNA {profile.character_uuid}", profile=item)
            else:
                output.stdout.write(
                    f"{profile.display_name}\nID: {profile.character_uuid}\n"
                    f"Version: {profile.profile_version}\n"
                    f"Hash: {profile.content_hash}\n"
                )
    except CLIError:
        raise
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise CLIError(
            f"CineDNA operation failed: {error}", code=ExitCode.INPUT
        ) from error


class _QuietOutput(Output):
    """Suppress successful intermediate stages of a composite command."""

    def success(self, message: str, **details: Any) -> None:
        pass


class _PreviewRenderer(BaseRenderer):
    """Deterministic, CPU-only renderer used for pipeline inspection."""

    def __init__(self, output_dir: Path, resolution: Resolution, fps: float) -> None:
        self.output_dir = output_dir
        self.resolution = resolution
        self.fps = fps

    @property
    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities(
            supported_resolution=(self.resolution,),
            supported_duration=Range(0, float("inf")),
            supported_fps=(self.fps,),
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


class _PreviewRendererPlugin(Plugin):
    """Built-in plugin that contributes the CPU-only preview renderer."""

    metadata = PluginMetadata(
        "preview-renderer", "1.0.0", "Built-in deterministic preview renderer"
    )

    def __init__(
        self, output_dir: Path, registry: RendererRegistry, package: Any
    ) -> None:
        self.output_dir = output_dir
        self.registry = registry
        self.resolution = Resolution(*package.project_metadata["resolution"])
        self.fps = package.project_metadata["fps"]

    def activate(self, context: PluginContext) -> None:
        if context.services["renderer_registry"] is not self.registry:
            raise ValueError(
                "preview plugin received an incompatible renderer registry"
            )
        self.registry.register(
            "preview",
            lambda: _PreviewRenderer(self.output_dir, self.resolution, self.fps),
        )
