"""Argument parsing and process boundary for the ``cineos`` executable."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import commands
from .errors import CLIError, ExitCode
from .output import Output


class _ArgumentParser(argparse.ArgumentParser):
    """Report usage failures through the CLI's normal output boundary."""

    def error(self, message: str) -> None:
        raise CLIError(
            f"invalid command usage: {message}",
            code=ExitCode.USAGE,
            hint=f"Run '{self.prog} --help' for usage and examples.",
        )


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="cineos",
        description="Compile and preview deterministic CINEOS film projects.",
        epilog="Example: cineos compile project.json --output film-package.json",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit machine-readable JSON output"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="validate a core project JSON file",
        epilog="Example: cineos validate project.json",
    )
    validate.add_argument("project", type=Path)
    compile_parser = subparsers.add_parser(
        "compile",
        help="compile a project into a Film Package",
        epilog="Example: cineos compile project.json --output film-package.json",
    )
    compile_parser.add_argument("project", type=Path)
    compile_parser.add_argument("--output", required=True, type=Path, metavar="FILE")
    render = subparsers.add_parser(
        "render",
        help="render a Film Package with the deterministic preview renderer",
        epilog="Example: cineos render film-package.json --output-dir renders",
    )
    render.add_argument("package", type=Path)
    render.add_argument("--output-dir", required=True, type=Path)
    assemble = subparsers.add_parser(
        "assemble",
        help="assemble preview renders into a preview movie",
        epilog="Example: cineos assemble renders --output movie.mp4",
    )
    assemble.add_argument("render_directory", type=Path)
    assemble.add_argument("--output", required=True, type=Path, metavar="FILE")
    demo = subparsers.add_parser(
        "demo",
        help="run the integrated preview pipeline",
        epilog="Example: cineos demo --output-dir demo-output",
    )
    demo.add_argument("--output-dir", required=True, type=Path)
    hardware = subparsers.add_parser(
        "hardware-report", help="inspect hardware for safe local rendering"
    )
    hardware.add_argument("--output", type=Path, metavar="FILE")
    hardware.add_argument(
        "--verbose", action="store_true", help="include raw diagnostic data"
    )
    assets = subparsers.add_parser("assets", help="manage a CINEOS asset registry")
    assets.add_argument(
        "--registry", type=Path, default=Path("assets.json"), help="asset registry JSON"
    )
    asset_commands = assets.add_subparsers(dest="asset_command", required=True)
    assets_list = asset_commands.add_parser("list", help="list registered assets")
    assets_list.add_argument("registry_path", type=Path, nargs="?")
    assets_show = asset_commands.add_parser("show", help="show one registered asset")
    assets_show.add_argument("asset_id")
    for command in ("add-character", "add-environment"):
        add = asset_commands.add_parser(
            command, help="register an asset from a manifest"
        )
        add.add_argument("manifest", type=Path)
    assets_validate = asset_commands.add_parser(
        "validate", help="validate assets and relationships"
    )
    assets_validate.add_argument("registry_path", type=Path, nargs="?")
    assets_export = asset_commands.add_parser(
        "export", help="export canonical asset JSON"
    )
    assets_export.add_argument("registry_path", type=Path, nargs="?")
    assets_export.add_argument("--output", required=True, type=Path, metavar="FILE")
    for asset_parser in asset_commands.choices.values():
        asset_parser.add_argument(
            "--json",
            action="store_true",
            default=argparse.SUPPRESS,
            help="emit machine-readable JSON output",
        )
    cinedna = subparsers.add_parser("cinedna", help="manage CineDNA identity profiles")
    cinedna.add_argument("--registry", type=Path, default=Path("assets.json"))
    cinedna.add_argument("--profiles", type=Path, default=Path("cinedna.json"))
    cinedna_commands = cinedna.add_subparsers(dest="cinedna_command", required=True)
    cinedna_build = cinedna_commands.add_parser(
        "build", help="build from a character asset"
    )
    cinedna_build.add_argument("character_id")
    cinedna_commands.add_parser("list", help="list identity profiles")
    cinedna_show = cinedna_commands.add_parser("show", help="show an identity profile")
    cinedna_show.add_argument("character_id")
    cinedna_validate = cinedna_commands.add_parser(
        "validate", help="validate an identity profile"
    )
    cinedna_validate.add_argument("character_id")
    cinedna_export = cinedna_commands.add_parser(
        "export", help="export an identity profile"
    )
    cinedna_export.add_argument("character_id")
    cinedna_export.add_argument("--output", required=True, type=Path)
    for cinedna_parser in cinedna_commands.choices.values():
        cinedna_parser.add_argument(
            "--json",
            action="store_true",
            default=argparse.SUPPRESS,
            help="emit machine-readable JSON output",
        )
    condition = subparsers.add_parser(
        "condition", help="build and inspect renderer-independent conditioning"
    )
    condition.add_argument("--package", type=Path, default=Path("film-package.json"))
    condition.add_argument("--registry", type=Path, default=Path("assets.json"))
    condition.add_argument("--profiles", type=Path, default=Path("cinedna.json"))
    condition_commands = condition.add_subparsers(
        dest="condition_command", required=True
    )
    for name in ("build", "export"):
        item = condition_commands.add_parser(
            name, help=f"{name} conditioning for one shot"
        )
        item.add_argument("shot_id")
        if name == "export":
            item.add_argument("--output", required=True, type=Path)
    for name in ("validate", "show"):
        item = condition_commands.add_parser(
            name, help=f"{name} a conditioning package"
        )
        item.add_argument("conditioning_package", type=Path)
    renderer = subparsers.add_parser("renderer", help="inspect and run Atlas renderers")
    renderer_commands = renderer.add_subparsers(dest="renderer_command", required=True)
    renderer_commands.add_parser("list", help="list installed renderers")
    for name in ("inspect", "validate"):
        item = renderer_commands.add_parser(name, help=f"{name} a renderer")
        item.add_argument("renderer_id")
        item.add_argument("--config", type=Path)
    renderer_render = renderer_commands.add_parser(
        "render", help="render one compiled shot"
    )
    renderer_render.add_argument("film_package", type=Path)
    renderer_render.add_argument("--renderer", dest="renderer_id", default="local-ai")
    renderer_render.add_argument("--shot", required=True)
    renderer_render.add_argument("--conditioning", required=True, type=Path)
    renderer_render.add_argument("--output", required=True, type=Path)
    renderer_render.add_argument("--config", type=Path)
    renderer_render.add_argument("--dry-run", action="store_true")
    validate_render = subparsers.add_parser(
        "validate-render", help="validate a completed rendered shot"
    )
    validate_render.add_argument("render", type=Path)
    validate_render.add_argument("--shot", required=True)
    validate_render.add_argument("--conditioning", required=True, type=Path)
    validate_render.add_argument("--output", required=True, type=Path)
    validation = subparsers.add_parser(
        "validation", help="inspect and compare render validation"
    )
    validation_commands = validation.add_subparsers(
        dest="validation_command", required=True
    )
    validation_show = validation_commands.add_parser("show")
    validation_show.add_argument("report", type=Path)
    validation_compare = validation_commands.add_parser("compare")
    validation_compare.add_argument("previous", type=Path)
    validation_compare.add_argument("current", type=Path)
    film = subparsers.add_parser("film", help="build and manage complete short films")
    film_commands = film.add_subparsers(dest="film_command", required=True)
    film_build = film_commands.add_parser("build")
    film_build.add_argument("project", type=Path)
    film_build.add_argument("--renderer", required=True, dest="renderer_id")
    film_build.add_argument("--output-dir", required=True, type=Path)
    film_build.add_argument("--dry-run", action="store_true")
    film_build.add_argument("--max-parallel-shots", type=int, default=1)
    film_build.add_argument("--max-recovery-attempts", type=int, default=1)
    film_build.add_argument("--skip-audio", action="store_true")
    film_build.add_argument("--skip-subtitles", action="store_true")
    film_build.add_argument("--manual-review-on-failure", action="store_true")
    film_build.add_argument("--resume", action="store_true")
    film_status = film_commands.add_parser("status")
    film_status.add_argument("build", type=Path)
    film_resume = film_commands.add_parser("resume")
    film_resume.add_argument("build", type=Path)
    film_cancel = film_commands.add_parser("cancel")
    film_cancel.add_argument("build_id")
    film_export = film_commands.add_parser("export")
    film_export.add_argument("build", type=Path)
    film_export.add_argument("--output", required=True, type=Path)
    subparsers.add_parser("version", help="print the installed CINEOS version")
    for command_parser in subparsers.choices.values():
        command_parser.add_argument(
            "--json",
            action="store_true",
            default=argparse.SUPPRESS,
            help="emit machine-readable JSON output",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    output = Output(json_mode="--json" in arguments)
    try:
        args = _parser().parse_args(arguments)
        output.json_mode = args.json
        if args.command == "validate":
            commands.validate(args.project, output)
        elif args.command == "compile":
            commands.compile(args.project, args.output, output)
        elif args.command == "render":
            commands.render(args.package, args.output_dir, output)
        elif args.command == "assemble":
            commands.assemble(args.render_directory, args.output, output)
        elif args.command == "demo":
            commands.demo(args.output_dir, output)
        elif args.command == "hardware-report":
            commands.hardware_report(args.output, args.verbose, output)
        elif args.command == "assets":
            commands.assets(
                args.asset_command,
                getattr(args, "registry_path", None) or args.registry,
                output,
                getattr(args, "output", None),
                manifest=getattr(args, "manifest", None),
                asset_id=getattr(args, "asset_id", None),
            )
        elif args.command == "cinedna":
            commands.cinedna(
                args.cinedna_command,
                args.registry,
                args.profiles,
                output,
                character_id=getattr(args, "character_id", None),
                destination=getattr(args, "output", None),
            )
        elif args.command == "condition":
            commands.condition(
                args.condition_command,
                output,
                package_path=args.package,
                registry_path=args.registry,
                profiles_path=args.profiles,
                shot_id=getattr(args, "shot_id", None),
                conditioning_path=getattr(args, "conditioning_package", None),
                destination=getattr(args, "output", None),
            )
        elif args.command == "renderer":
            commands.renderer(
                args.renderer_command,
                output,
                renderer_id=getattr(args, "renderer_id", None),
                config_path=getattr(args, "config", None),
                package_path=getattr(args, "film_package", None),
                conditioning_path=getattr(args, "conditioning", None),
                shot_id=getattr(args, "shot", None),
                destination=getattr(args, "output", None),
                dry_run=getattr(args, "dry_run", False),
            )
        elif args.command == "validate-render":
            commands.validate_render(
                args.render, args.shot, args.conditioning, args.output, output
            )
        elif args.command == "validation":
            commands.validation(
                args.validation_command,
                output,
                report_path=getattr(args, "report", None),
                previous=getattr(args, "previous", None),
                current=getattr(args, "current", None),
            )
        elif args.command == "film":
            commands.film(
                args.film_command,
                output,
                project=getattr(args, "project", None),
                build_path=getattr(args, "build", None),
                build_id=getattr(args, "build_id", None),
                renderer_id=getattr(args, "renderer_id", None),
                output_dir=getattr(args, "output_dir", None),
                destination=getattr(args, "output", None),
                dry_run=getattr(args, "dry_run", False),
            )
        else:
            commands.version(output)
    except CLIError as error:
        output.error(str(error), code=int(error.code), hint=error.hint)
        return int(error.code)
    except Exception as error:
        output.error(
            f"unexpected execution failure: {error}",
            code=int(ExitCode.EXECUTION),
            hint="Run again with valid inputs; report persistent failures.",
        )
        return int(ExitCode.EXECUTION)
    return int(ExitCode.SUCCESS)


if __name__ == "__main__":
    raise SystemExit(main())
