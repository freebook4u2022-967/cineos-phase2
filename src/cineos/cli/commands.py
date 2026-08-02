"""Thin command adapters that compose existing CINEOS APIs."""

from __future__ import annotations

import json
from dataclasses import asdict
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
from cineos.audio import (
    AudioExporter,
    AudioValidator,
    LipSyncMetadata,
    Mixer,
    MixInput,
    ProviderRegistry,
)
from cineos.audio.planner import plan_audio
from cineos.audio.serializer import load as load_audio
from cineos.audio.serializer import save as save_audio
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
from cineos.film import FilmBuild
from cineos.film.serializer import build_to_dict
from cineos.film.serializer import load as load_build
from cineos.film.serializer import save as save_build
from cineos.hardware import probe as probe_hardware
from cineos.hardware import to_json as hardware_to_json
from cineos.hardware import to_text as hardware_to_text
from cineos.nova import (
    CreativeBrief,
    CritiqueFinding,
    NOVACritic,
    NOVADirector,
    NOVARevisionEngine,
)
from cineos.nova.serializer import load as load_nova
from cineos.nova.serializer import save as save_nova
from cineos.plugins import Plugin, PluginContext, PluginManager, PluginMetadata
from cineos.renderers.local_ai import LocalAIConfig, LocalAIRendererPlugin
from cineos.renderers.local_ai.installer import setup_commands
from cineos.validation import (
    FakeValidatorBackend,
    TemporalValidator,
    ValidationPipeline,
)
from cineos.validation.serializer import load as load_validation
from cineos.validation.serializer import report_to_dict
from cineos.validation.serializer import save as save_validation

from .errors import CLIError, ExitCode
from .output import Output


def audio(
    command: str,
    output: Output,
    *,
    source: Path,
    destination: Path | None = None,
    output_dir: Path | None = None,
    language: str | None = None,
    provider_id: str | None = None,
    dry_run: bool = False,
    skip_dialogue: bool = False,
    skip_music: bool = False,
    skip_effects: bool = False,
    normalize_target: float | None = None,
    output_format: str = "wav",
) -> None:
    """Plan, validate, synthesize, mix, and export audio deliverables."""
    if command == "plan":
        movie = load_project(source)
        package = compile_project(movie)
        project = plan_audio(
            movie, package.content_hashes.get("package", ""), language=language or "en"
        )
        if not dry_run:
            assert destination
            save_audio(project, destination)
        output.success(
            (
                "Audio project validated"
                if dry_run
                else f"Audio project written to {destination}"
            ),
            project_id=project.project_id,
            content_hash=project.content_hash,
            cues=len(project.dialogue_tracks),
            dry_run=dry_run,
        )
        return
    project = load_audio(source)
    if language:
        project.language = language
    if normalize_target is not None:
        project.mix_settings.normalization_target = normalize_target
    registry = ProviderRegistry()
    provider = registry.get(provider_id) if provider_id else None
    report = AudioValidator().validate(project, provider=provider, check_ffmpeg=True)
    if command in {"cast", "inspect"} or dry_run:
        output.success(
            "Audio preflight complete",
            valid=report.valid,
            errors=report.errors,
            warnings=report.warnings,
            ffmpeg_available=report.ffmpeg_available,
            expected_outputs=report.expected_outputs,
            dry_run=dry_run,
        )
        return
    if command == "synthesize":
        if skip_dialogue:
            output.success("Dialogue synthesis skipped", synthesized=0)
            return
        if provider is None:
            raise ValueError("--provider is required for synthesis")
        assert output_dir
        profiles = {item.voice_id: item for item in project.voice_profiles}
        count = 0
        for cue in project.dialogue_tracks:
            voice_id = cue.approved_voice_profile_id or project.voice_assignments.get(
                cue.character_id
            )
            if voice_id not in profiles:
                raise ValueError(f"missing approved voice for cue {cue.cue_id}")
            result = provider.synthesize(
                cue, profiles[voice_id], output_dir / f"{cue.cue_id}.wav"
            )
            project.lip_sync_metadata.append(
                LipSyncMetadata(
                    cue.shot_id,
                    cue.character_id,
                    cue.cue_id,
                    result.phonemes,
                    result.words,
                    source_provider=result.provider_id,
                    timing_confidence=1.0,
                )
            )
            count += 1
        output.success(
            f"Synthesized dialogue to {output_dir}",
            synthesized=count,
            provider=provider.provider_id,
        )
    elif command == "mix":
        assert destination
        assets = source.parent / "synthesis"
        tracks = (
            [
                MixInput(
                    assets / f"{cue.cue_id}.wav",
                    cue.start_time,
                    cue.gain,
                    "dialogue",
                    cue.fade_in,
                    cue.fade_out,
                    cue.pan,
                    cue.muted,
                )
                for cue in project.dialogue_tracks
            ]
            if not skip_dialogue
            else []
        )
        result = Mixer(
            project.sample_rate, project.channel_layout, project.mix_settings
        ).mix(tracks, destination)
        if output_format != "wav":
            result = Mixer(
                project.sample_rate, project.channel_layout, project.mix_settings
            ).convert(result, result.with_suffix(f".{output_format}"))
        output.success(
            f"Mixed audio written to {result}",
            silence_fallback=not any(item.path.is_file() for item in tracks),
        )
    elif command == "export":
        assert output_dir
        outputs = AudioExporter().export(project, output_dir)
        output.success(
            f"Audio deliverables written to {output_dir}",
            outputs={key: str(value) for key, value in outputs.items()},
        )


def nova(
    command: str,
    output: Output,
    *,
    source: Path,
    destination: Path | None = None,
    critique_path: Path | None = None,
    seed: int = 0,
    planner: str = "rule-based",
    max_scenes: int | None = None,
    max_shots: int | None = None,
    target_duration: float | None = None,
    dry_run: bool = False,
) -> None:
    """Plan, critique, revise, or inspect a NOVA project."""
    if command == "plan":
        raw = _read_json(source)
        registry_path = raw.pop("asset_registry", None)
        registry = (
            load_assets(source.parent / registry_path)
            if registry_path
            else ProductionAssetRegistry()
        )
        brief = CreativeBrief(**raw)
        if target_duration is not None:
            brief.target_duration = target_duration
        plan = NOVADirector(registry).create_plan(
            brief,
            seed=seed,
            planner=planner,
            max_scenes=max_scenes,
            max_shots=max_shots,
        )
        if not dry_run:
            assert destination
            save_nova(plan, destination)
        output.success(
            "NOVA plan validated" if dry_run else f"NOVA plan written to {destination}",
            scenes=len(plan.scenes),
            shots=len(plan.shots),
            story_hash=plan.story.content_hash,
            dry_run=dry_run,
        )
        return
    plan = load_nova(source)
    if command == "critique":
        findings = NOVACritic().critique(plan)
        assert destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps([asdict(item) for item in findings], indent=2) + "\n",
            encoding="utf-8",
        )
        output.success(f"Critique written to {destination}", findings=len(findings))
    elif command == "revise":
        assert critique_path and destination
        values = json.loads(critique_path.read_text(encoding="utf-8"))
        findings = [CritiqueFinding(**item) for item in values]
        revised = NOVARevisionEngine().revise(plan, findings)
        save_nova(revised, destination)
        output.success(f"Revised plan written to {destination}", findings=len(findings))
    elif command == "show":
        output.success(
            f"{plan.brief.title}: {len(plan.scenes)} scenes, {len(plan.shots)} shots",
            title=plan.brief.title,
            logline=plan.story.logline,
            duration=sum(item.duration for item in plan.shots),
            scenes=len(plan.scenes),
            shots=len(plan.shots),
            planner=plan.planner_id,
        )


def film(
    command: str,
    output: Output,
    *,
    project: Path | None = None,
    build_path: Path | None = None,
    build_id: str | None = None,
    renderer_id: str | None = None,
    output_dir: Path | None = None,
    destination: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Manage persisted complete-film builds."""
    if command == "build":
        assert project and output_dir and renderer_id
        package = compile_project(load_project(project))
        package_id = package.content_hashes.get("package", "")
        build = FilmBuild(project.stem, package_id, renderer_id)
        build.metadata["plan"] = {
            "shot_count": len(package.shot_manifest),
            "assets": len(package.asset_manifest),
            "cinedna": len(package.cinedna_ids),
            "expected_output": str(output_dir),
        }
        if dry_run:
            build.metadata["dry_run"] = True
        else:
            raise CLIError(
                "renderer execution requires an application-registered film backend; "
                "use --dry-run to preflight",
                code=ExitCode.EXECUTION,
            )
        path = save_build(build, output_dir / "build.json")
        output.success(f"Film build plan written to {path}", build=build_to_dict(build))
    elif command == "status":
        assert build_path
        build = load_build(build_path)
        output.success(
            f"Build {build.build_id}: {build.status}", build=build_to_dict(build)
        )
    elif command == "resume":
        assert build_path
        build = load_build(build_path)
        build.metadata["resume_requested"] = True
        save_build(build, build_path)
        output.success(
            f"Resume recorded for build {build.build_id}", build=build_to_dict(build)
        )
    elif command == "cancel":
        # Build IDs are resolved by the caller's build store; no global mutable store
        # is invented by this local CLI.
        output.success(
            f"Cancellation requested for build {build_id}", build_id=build_id
        )
    elif command == "export":
        assert build_path and destination
        build = load_build(build_path)
        source = build.output_files.get("final_mp4")
        if not source or not Path(source).is_file():
            raise CLIError("build has no valid final MP4", code=ExitCode.EXECUTION)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(source).read_bytes())
        output.success(f"Film exported to {destination}", output=str(destination))


def validate_render(
    render_path: Path,
    shot_id: str,
    conditioning_path: Path,
    destination: Path,
    output: Output,
) -> None:
    """Validate one completed render and persist its structured report."""
    try:
        conditioning = load_conditioning(conditioning_path)
        report = ValidationPipeline().validate(
            render_path,
            conditioning,
            shot_id=shot_id,
            renderer_id="cli",
        )
        save_validation(report, destination)
    except (OSError, ValueError, RuntimeError) as error:
        raise CLIError(
            f"render validation failed: {error}", code=ExitCode.VALIDATION
        ) from error
    output.success(
        f"Validation report written to {destination}",
        report=report_to_dict(report),
    )


def validation(
    command: str,
    output: Output,
    *,
    report_path: Path | None,
    previous: Path | None,
    current: Path | None,
) -> None:
    """Show a report or compare temporal continuity between two renders."""
    if command == "show":
        if report_path is None:
            raise CLIError("report path is required", code=ExitCode.USAGE)
        report = load_validation(report_path)
        payload = report_to_dict(report)
        output.success(
            f"{report.shot_id}: {report.overall_status.value} "
            f"({report.overall_score})",
            report=payload,
        )
        return
    if previous is None or current is None:
        raise CLIError("two render paths are required", code=ExitCode.USAGE)
    if not previous.is_file() or not current.is_file():
        raise CLIError("comparison render does not exist", code=ExitCode.INPUT)
    # Plugins can replace these deterministic zero-drift metrics with optical
    # flow, perceptual hashes, or temporal models without changing the CLI.
    backend = FakeValidatorBackend(temporal={})
    conditioning = {
        "character_conditioning": [],
        "wardrobe_conditioning": [],
        "prop_conditioning": [],
        "environment_conditioning": None,
    }
    report = ValidationPipeline(backend, validators=[TemporalValidator()]).validate(
        current,
        conditioning,
        shot_id=current.stem,
        renderer_id="comparison",
        frames=[previous, current],
    )
    output.success(
        f"Compared {previous} with {current}", comparison=report_to_dict(report)
    )


def renderer(
    command: str,
    output: Output,
    *,
    renderer_id: str | None,
    config_path: Path | None,
    package_path: Path | None,
    conditioning_path: Path | None,
    shot_id: str | None,
    destination: Path | None,
    dry_run: bool,
) -> None:
    """Inspect, validate, or execute the built-in Atlas renderer plugin."""
    if command == "list":
        output.success(
            "local-ai 1.0.0", renderers=[{"id": "local-ai", "version": "1.0.0"}]
        )
        return
    if renderer_id != "local-ai":
        raise CLIError(f"unknown renderer: {renderer_id}", code=ExitCode.INPUT)
    try:
        config = LocalAIConfig.load(config_path)
        plugin = LocalAIRendererPlugin(config)
        if command == "inspect":
            caps = plugin.renderer.capabilities
            output.success(
                "local-ai: damo-vilab/text-to-video-ms-1.7b",
                renderer={
                    "id": "local-ai",
                    "version": plugin.version,
                    "model": plugin.renderer.model_identifier,
                    "config": config.to_dict(),
                    "resolutions": [
                        [item.width, item.height] for item in caps.supported_resolution
                    ],
                    "duration": [
                        caps.supported_duration.minimum,
                        caps.supported_duration.maximum,
                    ],
                    "fps": list(caps.supported_fps),
                    "features": sorted(caps.supported_features),
                    "setup_commands": list(setup_commands()),
                },
            )
            return
        if command == "validate":
            report = plugin.renderer.validate_environment()
            if not report.valid:
                raise CLIError(
                    "renderer environment is invalid: " + "; ".join(report.errors),
                    code=ExitCode.VALIDATION,
                    hint="See docs/RENDERER_SETUP.md; no dependencies were installed.",
                )
            output.success(
                "local-ai environment is valid",
                warnings=report.warnings,
                details=report.details,
            )
            return
        if not all((package_path, conditioning_path, shot_id, destination)):
            raise CLIError(
                "renderer render requires package, conditioning, shot, and output",
                code=ExitCode.USAGE,
            )
        package = load_package(package_path)
        conditioning = load_conditioning(conditioning_path)
        result = plugin.render(
            package, conditioning, destination, shot_id=shot_id, dry_run=dry_run
        )
        if dry_run:
            output.success(
                "renderer dry-run passed",
                request={
                    "job_id": result.job_id,
                    "shot_id": result.shot_id,
                    "prompt": result.prompt,
                    "seed": result.seed,
                    "output": str(result.output_path),
                    "resolution": [result.width, result.height],
                    "fps": result.fps,
                    "duration": result.duration,
                },
            )
        else:
            output.success(
                f"rendered {shot_id} to {destination}", result=result.to_dict()
            )
    except CLIError:
        raise
    except (OSError, ValueError, RuntimeError) as error:
        raise CLIError(
            f"renderer operation failed: {error}",
            code=ExitCode.VALIDATION,
            hint="Inspect the request and run 'cineos renderer validate local-ai'.",
        ) from error


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
    # Studio persists the canonical model under ``project`` while the original
    # CLI format is flat. Both representations describe the same MovieProject.
    if isinstance(value.get("project"), dict):
        value = value["project"]
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
