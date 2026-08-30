"""Persistent multi-shot Wan2.2 GPU validation for CINEOS.

The underlying Wan2.2 checkpoint remains an external pretrained foundation.
This module validates CINEOS-owned sequence orchestration, approved-reference
conditioning, continuity metadata, artifact integrity, persistent model reuse,
and optional QC-driven reject/rerender across a production-style 5-10 shot
benchmark.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .foundation_profiles import WAN22_TI2V_5B_PROFILE
from .wan22_execution import (
    Wan22ExecutionConfig,
    Wan22ExecutionError,
    _validate_artifact,
    build_wan22_execution_request,
)


@dataclass(frozen=True, slots=True)
class Wan22SequenceShot:
    """One connected shot in a persistent Wan2.2 benchmark sequence."""

    shot_id: str
    scene_id: str
    config: Wan22ExecutionConfig
    continuity_note: str = ""

    def __post_init__(self) -> None:
        if not self.shot_id.strip():
            raise ValueError("shot_id must not be blank")
        if not self.scene_id.strip():
            raise ValueError("scene_id must not be blank")


def _sequence_digest(receipts: Sequence[dict[str, Any]]) -> str:
    evidence = [
        {
            "shot_id": item["shot_id"],
            "request_hash": item["request_hash"],
            "artifact_sha256": item["artifact"]["sha256"],
        }
        for item in receipts
    ]
    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_sequence_contract(
    shots: Sequence[Wan22SequenceShot],
    *,
    require_shared_reference: bool,
) -> str | None:
    if not 5 <= len(shots) <= 10:
        raise Wan22ExecutionError(
            "production sequence validation requires between 5 and 10 connected shots"
        )

    shot_ids = [shot.shot_id for shot in shots]
    if len(set(shot_ids)) != len(shot_ids):
        raise Wan22ExecutionError("sequence shot_id values must be unique")

    references = [shot.config.approved_reference_id for shot in shots]
    if require_shared_reference:
        if any(reference is None for reference in references):
            raise Wan22ExecutionError(
                "identity sequence validation requires an approved reference on every shot"
            )
        unique = {reference for reference in references if reference is not None}
        if len(unique) != 1:
            raise Wan22ExecutionError(
                "identity sequence validation requires one shared approved reference"
            )
        return next(iter(unique))

    used = {reference for reference in references if reference is not None}
    return next(iter(used)) if len(used) == 1 else None


def _quality_report(
    evaluator: Any | None,
    *,
    output_path: str,
    shot: Wan22SequenceShot,
    attempt_index: int,
) -> dict[str, Any]:
    if evaluator is None:
        return {"accepted": True, "decision": "not_evaluated"}

    report = evaluator(
        output_path,
        shot=shot,
        attempt_index=attempt_index,
    )
    if not isinstance(report, dict):
        raise Wan22ExecutionError("shot_quality_evaluator must return a dict")
    accepted = report.get("accepted")
    if not isinstance(accepted, bool):
        raise Wan22ExecutionError(
            "shot_quality_evaluator report must contain boolean 'accepted'"
        )
    normalized = dict(report)
    normalized.setdefault("decision", "accept" if accepted else "reject")
    return normalized


def _preserve_rejected_artifact(output_path: str, *, attempt_index: int) -> str:
    path = Path(output_path)
    rejected_path = path.with_name(
        f"{path.stem}.rejected-{attempt_index + 1:02d}{path.suffix}"
    )
    if rejected_path.exists():
        rejected_path.unlink()
    path.replace(rejected_path)
    return str(rejected_path)


def run_wan22_gpu_sequence_validation(
    shots: Sequence[Wan22SequenceShot],
    *,
    output_dir: str | Path,
    device: str = "cuda",
    memory_strategy: str = "model_cpu_offload",
    dtype: str = "bfloat16",
    enable_vae_tiling: bool = True,
    require_shared_reference: bool = True,
    reference_loader: Any | None = None,
    pipeline_factory: Any | None = None,
    video_exporter: Any | None = None,
    shot_quality_evaluator: Any | None = None,
    max_rerender_attempts: int = 2,
) -> dict[str, Any]:
    """Render and audit a connected 5-10 shot sequence with one loaded model.

    The renderer is initialized, loaded, and warmed exactly once, then reused for
    all shots. This removes repeated foundation loading from the benchmark path and
    makes a connected-film validation materially closer to production inference.

    Identity benchmarks fail closed by default: every shot must declare the same
    approved reference and a reference loader must be supplied. Set
    ``require_shared_reference=False`` only for non-character sequence benchmarks.

    ``shot_quality_evaluator`` is the CINEOS-owned QC boundary around the external
    foundation. It receives ``(output_path, shot=..., attempt_index=...)`` and must
    return a dict containing boolean ``accepted`` plus any metrics/directives. A
    rejected shot is preserved as an auditable artifact, then rerendered in the
    same persistent model session with a deterministic incremented seed. The run
    fails closed if no candidate passes within ``max_rerender_attempts`` retries.
    """

    if max_rerender_attempts < 0:
        raise ValueError("max_rerender_attempts must be non-negative")

    shared_reference_id = _validate_sequence_contract(
        shots,
        require_shared_reference=require_shared_reference,
    )
    has_any_reference = any(
        shot.config.approved_reference_id is not None for shot in shots
    )
    if has_any_reference and reference_loader is None:
        raise Wan22ExecutionError(
            "approved reference conditioning requires a reference_loader"
        )

    output_root = Path(output_dir)
    renderer = WAN22_TI2V_5B_PROFILE.renderer(
        output_dir=output_root,
        reference_loader=reference_loader,
        pipeline_factory=pipeline_factory,
        video_exporter=video_exporter,
    )

    receipts: list[dict[str, Any]] = []
    total_rejected_candidates = 0
    sequence_started = time.perf_counter()
    renderer.initialize()
    try:
        renderer.load_model(
            device=device,
            dtype=dtype,
            memory_strategy=memory_strategy,
            enable_vae_tiling=enable_vae_tiling,
        )
        renderer.warmup()

        previous_shot_id: str | None = None
        for index, shot in enumerate(shots):
            attempt_receipts: list[dict[str, Any]] = []
            accepted_result = None
            accepted_request = None
            accepted_artifact = None
            accepted_elapsed = None
            accepted_quality = None

            for attempt_index in range(max_rerender_attempts + 1):
                attempt_config = replace(
                    shot.config,
                    seed=shot.config.seed + attempt_index,
                )
                request = build_wan22_execution_request(attempt_config)
                request.shot_id = shot.shot_id
                request.scene_id = shot.scene_id
                request.continuity = {
                    "mode": "connected-sequence-foundation-validation",
                    "sequence_index": index,
                    "previous_shot_id": previous_shot_id,
                    "continuity_note": shot.continuity_note,
                    "rerender_attempt": attempt_index,
                }
                request.metadata.update(
                    {
                        "sequence_index": index,
                        "sequence_shot_count": len(shots),
                        "shared_reference_required": require_shared_reference,
                        "rerender_attempt": attempt_index,
                    }
                )
                request.refresh_hash()

                shot_started = time.perf_counter()
                result = renderer.render(request)
                shot_elapsed = time.perf_counter() - shot_started
                artifact = _validate_artifact(
                    result.output_path,
                    actual_frame_count=result.frame_count,
                    expected_frame_count=request.metadata["execution_frame_count"],
                )
                quality = _quality_report(
                    shot_quality_evaluator,
                    output_path=result.output_path,
                    shot=shot,
                    attempt_index=attempt_index,
                )
                attempt_receipt = {
                    "attempt_index": attempt_index,
                    "seed": result.seed,
                    "request_hash": result.request_hash,
                    "output_path": result.output_path,
                    "artifact": artifact,
                    "quality": quality,
                    "execution_elapsed_seconds": shot_elapsed,
                }

                if quality["accepted"]:
                    attempt_receipt["status"] = "accepted"
                    attempt_receipts.append(attempt_receipt)
                    accepted_result = result
                    accepted_request = request
                    accepted_artifact = artifact
                    accepted_elapsed = shot_elapsed
                    accepted_quality = quality
                    break

                total_rejected_candidates += 1
                attempt_receipt["status"] = "rejected"
                attempt_receipt["output_path"] = _preserve_rejected_artifact(
                    result.output_path,
                    attempt_index=attempt_index,
                )
                attempt_receipts.append(attempt_receipt)

            if accepted_result is None:
                raise Wan22ExecutionError(
                    f"shot {shot.shot_id!r} failed quality gate after "
                    f"{max_rerender_attempts + 1} candidate(s)"
                )

            receipts.append(
                {
                    "status": "rendered",
                    "sequence_index": index,
                    "shot_id": accepted_result.shot_id,
                    "scene_id": accepted_result.scene_id,
                    "previous_shot_id": previous_shot_id,
                    "output_path": accepted_result.output_path,
                    "artifact": accepted_artifact,
                    "actual_frame_count": accepted_result.frame_count,
                    "execution_duration_seconds": accepted_request.metadata[
                        "execution_duration_seconds"
                    ],
                    "execution_elapsed_seconds": accepted_elapsed,
                    "fps": shot.config.fps,
                    "seed": accepted_result.seed,
                    "request_hash": accepted_result.request_hash,
                    "approved_reference_id": shot.config.approved_reference_id,
                    "quality": accepted_quality,
                    "rerender_count": len(attempt_receipts) - 1,
                    "attempts": attempt_receipts,
                }
            )
            previous_shot_id = shot.shot_id
    finally:
        renderer.shutdown()

    elapsed_seconds = time.perf_counter() - sequence_started
    digest = _sequence_digest(receipts)
    total_frames = sum(item["actual_frame_count"] for item in receipts)
    total_duration = sum(item["execution_duration_seconds"] for item in receipts)
    manifest = {
        "status": "rendered",
        "benchmark": "cineos-wan22-connected-sequence/0.2",
        "shot_count": len(receipts),
        "sequence_sha256": digest,
        "total_frame_count": total_frames,
        "total_execution_duration_seconds": total_duration,
        "sequence_elapsed_seconds": elapsed_seconds,
        "conditioning": {
            "require_shared_reference": require_shared_reference,
            "shared_approved_reference_id": shared_reference_id,
        },
        "quality_control": {
            "enabled": shot_quality_evaluator is not None,
            "max_rerender_attempts": max_rerender_attempts,
            "rejected_candidate_count": total_rejected_candidates,
            "accepted_shot_count": len(receipts),
        },
        "runtime": {
            "device": device,
            "dtype": dtype,
            "memory_strategy": memory_strategy,
            "vae_tiling": enable_vae_tiling,
            "persistent_model_session": True,
        },
        "foundation_profile": WAN22_TI2V_5B_PROFILE.snapshot(),
        "shots": receipts,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "wan22-sequence-receipt.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest


__all__ = [
    "Wan22SequenceShot",
    "run_wan22_gpu_sequence_validation",
]
