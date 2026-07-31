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
