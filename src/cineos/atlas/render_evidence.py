"""Cryptographic evidence for real Atlas video render artifacts.

A successful model call is not enough to claim a production render. This module
creates a compact, deterministic proof record only after a concrete non-empty
artifact exists on disk. The proof binds that artifact to the CINEOS request,
pretrained-foundation provenance, and runtime selection used for execution.

It intentionally does not infer that a pretrained foundation is CINEOS-native.
The foundation model identifier remains explicit in every proof record.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class RenderEvidenceError(RuntimeError):
    """Raised when a render cannot produce trustworthy execution evidence."""


@dataclass(frozen=True, slots=True)
class RenderEvidence:
    """Auditable proof that one concrete render artifact was produced."""

    schema: str
    artifact_path: str
    artifact_bytes: int
    artifact_sha256: str
    shot_id: str
    scene_id: str
    frame_count: int
    seed: int
    request_hash: str
    foundation_model_id: str
    foundation_revision: str | None
    foundation_license_id: str | None
    device: str
    dtype: str
    memory_strategy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash an artifact without loading a potentially large video into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def collect_render_evidence(
    *,
    artifact_path: str | Path,
    shot_id: str,
    scene_id: str,
    frame_count: int,
    seed: int,
    request_hash: str,
    foundation_model_id: str,
    foundation_revision: str | None,
    foundation_license_id: str | None,
    device: str,
    dtype: str,
    memory_strategy: str,
) -> RenderEvidence:
    """Validate and bind a concrete artifact to its execution provenance."""
    path = Path(artifact_path)
    if not path.is_file():
        raise RenderEvidenceError(f"render artifact does not exist: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise RenderEvidenceError(f"render artifact is empty: {path}")
    if frame_count <= 0:
        raise RenderEvidenceError("frame_count must be positive")
    if not request_hash:
        raise RenderEvidenceError("request_hash is required")
    if not foundation_model_id:
        raise RenderEvidenceError("foundation_model_id is required")
    if not device:
        raise RenderEvidenceError("device is required")
    if not dtype:
        raise RenderEvidenceError("dtype is required")
    if not memory_strategy:
        raise RenderEvidenceError("memory_strategy is required")

    return RenderEvidence(
        schema="cineos-render-evidence/0.1",
        artifact_path=str(path),
        artifact_bytes=size,
        artifact_sha256=sha256_file(path),
        shot_id=shot_id,
        scene_id=scene_id,
        frame_count=frame_count,
        seed=seed,
        request_hash=request_hash,
        foundation_model_id=foundation_model_id,
        foundation_revision=foundation_revision,
        foundation_license_id=foundation_license_id,
        device=device,
        dtype=dtype,
        memory_strategy=memory_strategy,
    )


def write_render_evidence(
    evidence: RenderEvidence,
    path: str | Path | None = None,
) -> Path:
    """Atomically persist evidence next to the render unless a path is supplied."""
    target = (
        Path(path)
        if path is not None
        else Path(evidence.artifact_path).with_suffix(".render-evidence.json")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    payload = json.dumps(evidence.to_dict(), sort_keys=True, indent=2) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(target)
    return target
