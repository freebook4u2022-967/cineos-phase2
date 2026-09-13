import hashlib
from pathlib import Path

import pytest

from cineos.atlas.connected_continuity_evidence import (
    ConnectedContinuityEvidenceError,
    production_visual_continuity_evidence,
    validate_connected_visual_continuity,
)
from cineos.atlas.foundation_profiles import WAN22_TI2V_5B_PROFILE
from cineos.atlas.gpu_foundation_smoke import GPUFoundationExecutionReceipt
from cineos.atlas.gpu_preflight import GPUExecutionPlan
from cineos.atlas.production_continuity_diffusers import VISUAL_CONTINUITY_SCHEMA
from cineos.atlas.production_diffusers import ProductionDiffusersVideoResult


def _plan() -> GPUExecutionPlan:
    return GPUExecutionPlan(
        device="cuda:0",
        dtype="bfloat16",
        memory_strategy="resident",
        enable_vae_tiling=False,
        enable_vae_slicing=False,
        enable_attention_slicing=False,
        estimated_model_vram_gb=24.0,
        observed_total_vram_gb=48.0,
        observed_free_vram_gb=40.0,
        fit_margin_gb=16.0,
    )


def _receipt(
    tmp_path: Path,
    index: int,
    *,
    previous: GPUFoundationExecutionReceipt | None = None,
) -> GPUFoundationExecutionReceipt:
    scene_id = "scene-connected"
    shot_id = f"shot-{index}"
    artifact = tmp_path / f"{scene_id}-{shot_id}.mp4"
    payload = f"video-payload-{index}".encode()
    artifact.write_bytes(payload)
    artifact_sha256 = hashlib.sha256(payload).hexdigest()
    request_hash = hashlib.sha256(f"request-{index}".encode()).hexdigest()

    if previous is None:
        provenance = {
            "schema": VISUAL_CONTINUITY_SCHEMA,
            "mode": "approved_reference_root",
            "scene_id": scene_id,
            "shot_id": shot_id,
            "previous_scene_id": None,
            "previous_shot_id": None,
            "predecessor_artifact_sha256": None,
            "predecessor_request_hash": None,
            "current_artifact_sha256": artifact_sha256,
            "current_request_hash": request_hash,
            "approved_reference_ids": ["lead-reference"],
            "in_memory_terminal_frame": False,
        }
    else:
        provenance = {
            "schema": VISUAL_CONTINUITY_SCHEMA,
            "mode": "predecessor_terminal_frame_lineage",
            "scene_id": scene_id,
            "shot_id": shot_id,
            "previous_scene_id": previous.result.scene_id,
            "previous_shot_id": previous.result.shot_id,
            "predecessor_artifact_sha256": previous.output_sha256,
            "predecessor_request_hash": previous.result.request_hash,
            "current_artifact_sha256": artifact_sha256,
            "current_request_hash": request_hash,
            "approved_reference_ids": ["lead-reference"],
            "in_memory_terminal_frame": True,
        }

    result = ProductionDiffusersVideoResult(
        shot_id=shot_id,
        scene_id=scene_id,
        output_path=str(artifact),
        frame_count=48,
        seed=1000 + index,
        foundation=WAN22_TI2V_5B_PROFILE.provenance,
        request_hash=request_hash,
        artifact_sha256=artifact_sha256,
        artifact_size_bytes=len(payload),
        conditioning_provenance=provenance,
    )
    return GPUFoundationExecutionReceipt(
        result=result,
        execution_plan=_plan(),
        profile_id=WAN22_TI2V_5B_PROFILE.profile_id,
        origin=WAN22_TI2V_5B_PROFILE.origin,
        output_bytes=len(payload),
        output_sha256=artifact_sha256,
        elapsed_seconds=0.25,
    )


def _sequence(tmp_path: Path, count: int = 5) -> list[GPUFoundationExecutionReceipt]:
    receipts = []
    for index in range(count):
        receipts.append(
            _receipt(tmp_path, index, previous=receipts[-1] if receipts else None)
        )
    return receipts


def test_accepts_exact_artifact_bound_terminal_frame_lineage(tmp_path):
    receipts = _sequence(tmp_path)

    evidence = validate_connected_visual_continuity(receipts)

    assert len(evidence) == 5
    assert evidence[0]["mode"] == "approved_reference_root"
    assert evidence[-1]["predecessor_artifact_sha256"] == receipts[-2].output_sha256
    assert production_visual_continuity_evidence(receipts) is True


def test_rejects_swapped_predecessor_artifact_binding(tmp_path):
    receipts = _sequence(tmp_path)
    provenance = receipts[2].result.conditioning_provenance
    assert provenance is not None
    provenance["predecessor_artifact_sha256"] = receipts[0].output_sha256

    with pytest.raises(
        ConnectedContinuityEvidenceError,
        match="different predecessor artifact",
    ):
        validate_connected_visual_continuity(receipts)

    assert production_visual_continuity_evidence(receipts) is False


def test_rejects_current_artifact_substitution(tmp_path):
    receipts = _sequence(tmp_path)
    provenance = receipts[3].result.conditioning_provenance
    assert provenance is not None
    provenance["current_artifact_sha256"] = receipts[0].output_sha256

    with pytest.raises(
        ConnectedContinuityEvidenceError,
        match="current artifact does not match",
    ):
        validate_connected_visual_continuity(receipts)


def test_rejects_prompt_order_without_terminal_frame_attestation(tmp_path):
    receipts = _sequence(tmp_path)
    provenance = receipts[1].result.conditioning_provenance
    assert provenance is not None
    provenance["in_memory_terminal_frame"] = False

    with pytest.raises(
        ConnectedContinuityEvidenceError,
        match="does not attest terminal-frame handoff",
    ):
        validate_connected_visual_continuity(receipts)


def test_rejects_non_immediate_predecessor_identity(tmp_path):
    receipts = _sequence(tmp_path)
    provenance = receipts[4].result.conditioning_provenance
    assert provenance is not None
    provenance["previous_shot_id"] = receipts[1].result.shot_id

    with pytest.raises(
        ConnectedContinuityEvidenceError,
        match="immediate predecessor shot",
    ):
        validate_connected_visual_continuity(receipts)


def test_rejects_missing_returned_conditioning_provenance(tmp_path):
    receipts = _sequence(tmp_path)
    result = receipts[2].result
    object.__setattr__(result, "conditioning_provenance", None)

    with pytest.raises(
        ConnectedContinuityEvidenceError,
        match="requires returned conditioning provenance",
    ):
        validate_connected_visual_continuity(receipts)
