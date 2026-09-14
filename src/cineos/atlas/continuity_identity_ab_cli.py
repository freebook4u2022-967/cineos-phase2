"""CLI for comparing two real connected-shot GPU benchmark receipts.

The command never renders video and never upgrades evidence. It consumes two
existing production-quality-gated benchmark JSON files and emits an auditable
CINEOS A/B decision for the experimental continuity + fresh-reference strategy.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .continuity_identity_ab import (
    ContinuityIdentityABError,
    evaluate_continuity_identity_ab,
)


class ContinuityIdentityABCLIError(RuntimeError):
    """Raised when A/B input files cannot be trusted or decoded."""


def _load_receipt(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ContinuityIdentityABCLIError(
            f"cannot read A/B benchmark receipt: {source}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ContinuityIdentityABCLIError(
            f"A/B benchmark receipt is not valid JSON: {source}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ContinuityIdentityABCLIError(
            f"A/B benchmark receipt must be a JSON object: {source}"
        )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare predecessor-only and continuity-identity-refresh production "
            "GPU benchmark receipts."
        )
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    parser.add_argument("--minimum-identity-gain", type=float, default=0.02)
    parser.add_argument(
        "--maximum-identity-shot-regression",
        type=float,
        default=0.03,
    )
    parser.add_argument("--maximum-temporal-regression", type=float, default=0.01)
    parser.add_argument("--maximum-motion-regression", type=float, default=0.01)
    parser.add_argument("--maximum-artifact-regression", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        decision = evaluate_continuity_identity_ab(
            _load_receipt(args.baseline),
            _load_receipt(args.candidate),
            minimum_identity_gain=args.minimum_identity_gain,
            maximum_identity_shot_regression=args.maximum_identity_shot_regression,
            maximum_temporal_regression=args.maximum_temporal_regression,
            maximum_motion_regression=args.maximum_motion_regression,
            maximum_artifact_regression=args.maximum_artifact_regression,
        )
    except ContinuityIdentityABError as exc:
        raise ContinuityIdentityABCLIError(str(exc)) from exc

    payload = json.dumps(decision.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if decision.promotable else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ContinuityIdentityABCLIError", "main"]
