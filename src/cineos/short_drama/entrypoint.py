"""Top-level CLI routing for the CINEOS Short Drama Agent."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from cineos.cli.errors import ExitCode
from cineos.cli.main import main as core_main
from cineos.cli.output import Output

from .models import DramaBrief
from .orchestrator import ShortDramaOrchestrator
from .integration import write_production_artifacts


def _is_drama_command(argv: Sequence[str]) -> bool:
    return "drama" in argv and next((item for item in argv if not item.startswith("-")), None) == "drama"


def _drama_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cineos",
        description="CINEOS cinematic production and Short Drama Agent.",
    )
    parser.add_argument("--json", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    drama = commands.add_parser("drama", help="create CINEOS short-drama projects")
    drama_commands = drama.add_subparsers(dest="drama_command", required=True)
    create = drama_commands.add_parser(
        "create",
        help="turn one premise into a drama plan and Film Package",
    )
    create.add_argument("premise")
    create.add_argument("--duration", type=int, default=180, dest="duration_seconds")
    create.add_argument("--genre", default="drama")
    create.add_argument("--tone", default="cinematic")
    create.add_argument("--output-dir", type=Path, default=Path("."))
    create.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="emit machine-readable JSON output",
    )
    return parser


def _run_drama(argv: Sequence[str]) -> int:
    try:
        args = _drama_parser().parse_args(list(argv))
    except SystemExit as error:
        return int(error.code)

    output = Output(json_mode=args.json)
    try:
        brief = DramaBrief(
            premise=args.premise,
            duration_seconds=args.duration_seconds,
            genre=args.genre,
            tone=args.tone,
        )
        plan = ShortDramaOrchestrator().plan(brief)
        artifacts = write_production_artifacts(plan, args.output_dir)
        output.success(
            "CINEOS Short Drama project created",
            premise=brief.premise,
            duration_seconds=brief.duration_seconds,
            genre=brief.genre,
            drama_package=str(artifacts["drama_package"]),
            asset_registry=str(artifacts["asset_registry"]),
            film_package=str(artifacts["film_package"]),
            continuity_status=plan.continuity["status"],
        )
    except Exception as error:
        output.error(
            f"short-drama creation failed: {error}",
            code=int(ExitCode.EXECUTION),
            hint="Check the premise and output directory, then try again.",
        )
        return int(ExitCode.EXECUTION)
    return int(ExitCode.SUCCESS)


def main(argv: Sequence[str] | None = None) -> int:
    """Route ``cineos drama`` locally and preserve every existing CLI command."""
    values = list(sys.argv[1:] if argv is None else argv)
    if _is_drama_command(values):
        return _run_drama(values)
    return core_main(values)


if __name__ == "__main__":
    raise SystemExit(main())
