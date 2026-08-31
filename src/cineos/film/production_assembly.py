"""Fail-closed assembly of production films from approved evidence-bound assets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .assembly import assemble
from .exceptions import AssemblyError
from .validator import file_hash

PRODUCTION_EVIDENCE_SCHEMA = "cineos-production-film-evidence/0.1"


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_bound_shot(
    record: Mapping[str, Any], *, index: int
) -> tuple[str, Path, str]:
    shot_id = str(record.get("shot_id") or "").strip()
    if not shot_id:
        raise AssemblyError(f"production shot {index} is missing shot_id")
    if record.get("accepted") is not True or record.get("decision") != "accept":
        raise AssemblyError(f"production shot {shot_id} is not QC-approved")
    if record.get("production_gpu_evidence") is not True:
        raise AssemblyError(f"production shot {shot_id} lacks real GPU evidence")

    output_path = Path(str(record.get("output_path") or "")).resolve()
    expected_hash = str(record.get("output_sha256") or "").strip().lower()
    if len(expected_hash) != 64:
        raise AssemblyError(
            f"production shot {shot_id} is missing a valid output SHA-256"
        )
    actual_hash = file_hash(output_path)
    if actual_hash != expected_hash:
        raise AssemblyError(
            f"production shot {shot_id} artifact hash does not match QC evidence"
        )

    evidence_hash = str(record.get("evidence_sha256") or "").strip().lower()
    if len(evidence_hash) != 64:
        raise AssemblyError(f"production shot {shot_id} is missing evidence SHA-256")
    return shot_id, output_path, evidence_hash


def assemble_production_film(
    shot_evidence: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    durations: Sequence[float] | None = None,
    audio_path: str | Path | None = None,
    audio_sha256: str | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble only the exact artifacts accepted by production GPU/QC evidence.

    This boundary intentionally does not infer trust from file names or renderer
    labels. Every shot must carry explicit production-GPU provenance, an accept
    decision, and the SHA-256 of both its rendered artifact and evidence record.
    The resulting manifest binds the ordered inputs and final artifact so later
    packaging cannot silently swap a post-QC shot or audio mix.
    """
    if not 5 <= len(shot_evidence) <= 10:
        raise AssemblyError("production connected-film assembly requires 5 to 10 shots")
    if durations is not None and len(durations) != len(shot_evidence):
        raise AssemblyError("duration count does not match production shot count")

    bound: list[tuple[str, Path, str]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(shot_evidence):
        item = _require_bound_shot(record, index=index)
        if item[0] in seen_ids:
            raise AssemblyError(f"duplicate production shot_id: {item[0]}")
        seen_ids.add(item[0])
        bound.append(item)

    audio: dict[str, Any] | None = None
    audio_source: Path | None = None
    if audio_path is not None:
        if not audio_sha256 or len(audio_sha256.strip()) != 64:
            raise AssemblyError("production audio requires an explicit SHA-256")
        audio_source = Path(audio_path).resolve()
        actual_audio_hash = file_hash(audio_source)
        if actual_audio_hash != audio_sha256.strip().lower():
            raise AssemblyError(
                "production audio artifact hash does not match supplied evidence"
            )
        audio = {"path": str(audio_source), "sha256": actual_audio_hash}
    elif audio_sha256 is not None:
        raise AssemblyError("audio SHA-256 was supplied without an audio artifact")

    destination = Path(output).resolve()
    movie = assemble(
        [path for _, path, _ in bound],
        destination,
        durations=list(durations) if durations is not None else None,
        audio_path=audio_source,
    )
    final_hash = file_hash(movie)

    manifest: dict[str, Any] = {
        "schema": PRODUCTION_EVIDENCE_SCHEMA,
        "shot_count": len(bound),
        "shots": [
            {
                "index": index,
                "shot_id": shot_id,
                "output_path": str(path),
                "output_sha256": file_hash(path),
                "evidence_sha256": evidence_hash,
            }
            for index, (shot_id, path, evidence_hash) in enumerate(bound)
        ],
        "audio": audio,
        "final_mp4": str(movie.resolve()),
        "final_mp4_sha256": final_hash,
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)

    target = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else destination.with_suffix(".production.json")
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = ["PRODUCTION_EVIDENCE_SCHEMA", "assemble_production_film"]
