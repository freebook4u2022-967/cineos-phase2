"""Fail-fast dependency gate for production difficult-case semantic QC.

The connected-film production path uses external pretrained foundations for specialist
quality measurement.  CINEOS must prove those dependencies are locally available and
pinned before expensive video inference begins; otherwise a benchmark could render for
hours and only then discover that anatomy/interaction/physics or dialogue lip-sync
cannot be measured honestly.

This module does not turn borrowed models into CINEOS-native capability.  It only
verifies the exact external Qwen2.5-VL visual-judge snapshot and LatentSync SyncNet
runtime selected by the production semantic-QC contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .latentsync_syncnet_scorer import LATENTSYNC_PINNED_REVISION
from .qwen25vl_semantic_judge import QWEN25VL_MODEL_ID, QWEN25VL_MODEL_REVISION

PRODUCTION_SEMANTIC_QC_DEPENDENCY_SCHEMA = (
    "cineos-production-semantic-qc-dependency-readiness/0.1"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class SemanticQCDependencyReport:
    """Auditable readiness result for the external specialist QC stack."""

    qwen_ready: bool
    latentsync_repository_ready: bool
    latentsync_checkpoint_ready: bool
    blockers: tuple[str, ...]
    qwen_snapshot: str
    latentsync_repository: str
    latentsync_checkpoint: str
    latentsync_checkpoint_sha256: str | None

    @property
    def ready(self) -> bool:
        return (
            self.qwen_ready
            and self.latentsync_repository_ready
            and self.latentsync_checkpoint_ready
            and not self.blockers
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PRODUCTION_SEMANTIC_QC_DEPENDENCY_SCHEMA,
            "ready": self.ready,
            "origin": "cineos-dependency-verification",
            "external_components": {
                "qwen25vl_visual_judge": {
                    "origin": "external_pretrained",
                    "model_id": QWEN25VL_MODEL_ID,
                    "revision": QWEN25VL_MODEL_REVISION,
                    "snapshot": self.qwen_snapshot,
                    "ready": self.qwen_ready,
                },
                "latentsync_syncnet": {
                    "origin": "external_pretrained",
                    "repository_revision": LATENTSYNC_PINNED_REVISION,
                    "repository": self.latentsync_repository,
                    "checkpoint": self.latentsync_checkpoint,
                    "checkpoint_sha256": self.latentsync_checkpoint_sha256,
                    "repository_ready": self.latentsync_repository_ready,
                    "checkpoint_ready": self.latentsync_checkpoint_ready,
                },
            },
            "blockers": list(self.blockers),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repository_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    observed = completed.stdout.strip()
    return observed if re.fullmatch(r"[0-9a-f]{40}", observed) else None


def _qwen_snapshot_ready(snapshot: Path) -> bool:
    """Require a revision-bound local snapshot with model and processor material."""

    if not snapshot.is_dir() or snapshot.name != QWEN25VL_MODEL_REVISION:
        return False
    if not (snapshot / "config.json").is_file():
        return False
    processor_present = any(
        (snapshot / name).is_file()
        for name in (
            "preprocessor_config.json",
            "processor_config.json",
            "tokenizer_config.json",
        )
    )
    if not processor_present:
        return False
    model_present = (snapshot / "model.safetensors.index.json").is_file() or any(
        snapshot.glob("*.safetensors")
    )
    return model_present


def evaluate_semantic_qc_dependencies(
    *,
    qwen_snapshot: str | Path,
    latentsync_repository: str | Path,
    latentsync_checkpoint: str | Path,
    latentsync_checkpoint_sha256: str,
) -> SemanticQCDependencyReport:
    """Verify exact local external-QC dependencies without loading model weights."""

    if not isinstance(latentsync_checkpoint_sha256, str) or not _SHA256_RE.fullmatch(
        latentsync_checkpoint_sha256
    ):
        raise ValueError(
            "latentsync_checkpoint_sha256 must be a lowercase SHA-256 digest"
        )

    qwen_path = Path(qwen_snapshot).expanduser().resolve()
    repository_path = Path(latentsync_repository).expanduser().resolve()
    checkpoint_path = Path(latentsync_checkpoint).expanduser().resolve()
    blockers: list[str] = []

    qwen_ready = _qwen_snapshot_ready(qwen_path)
    if not qwen_ready:
        blockers.append(
            "pinned Qwen2.5-VL specialist-QC snapshot is unavailable or incomplete"
        )

    observed_revision = _git_head(repository_path) if repository_path.is_dir() else None
    repository_ready = observed_revision == LATENTSYNC_PINNED_REVISION
    if not repository_ready:
        blockers.append(
            "LatentSync repository is unavailable or not at the pinned production revision"
        )

    observed_checkpoint_hash: str | None = None
    if checkpoint_path.is_file():
        try:
            observed_checkpoint_hash = _sha256(checkpoint_path)
        except OSError:
            observed_checkpoint_hash = None
    checkpoint_ready = observed_checkpoint_hash == latentsync_checkpoint_sha256
    if not checkpoint_ready:
        blockers.append(
            "LatentSync SyncNet checkpoint is unavailable or does not match approved bytes"
        )

    return SemanticQCDependencyReport(
        qwen_ready=qwen_ready,
        latentsync_repository_ready=repository_ready,
        latentsync_checkpoint_ready=checkpoint_ready,
        blockers=tuple(blockers),
        qwen_snapshot=str(qwen_path),
        latentsync_repository=str(repository_path),
        latentsync_checkpoint=str(checkpoint_path),
        latentsync_checkpoint_sha256=observed_checkpoint_hash,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail fast unless the exact external specialist semantic-QC dependencies "
            "required by the competitive connected-film benchmark are locally ready."
        )
    )
    parser.add_argument("--qwen-snapshot", required=True)
    parser.add_argument("--latentsync-repository", required=True)
    parser.add_argument("--latentsync-checkpoint", required=True)
    parser.add_argument("--latentsync-checkpoint-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate_semantic_qc_dependencies(
        qwen_snapshot=args.qwen_snapshot,
        latentsync_repository=args.latentsync_repository,
        latentsync_checkpoint=args.latentsync_checkpoint,
        latentsync_checkpoint_sha256=args.latentsync_checkpoint_sha256,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PRODUCTION_SEMANTIC_QC_DEPENDENCY_SCHEMA",
    "SemanticQCDependencyReport",
    "evaluate_semantic_qc_dependencies",
    "main",
]
