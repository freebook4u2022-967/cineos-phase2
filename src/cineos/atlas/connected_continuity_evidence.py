"""Artifact-bound validation for connected visual-continuity production evidence.

CINEOS owns the continuity orchestration and evidence contract in this module. The
underlying pretrained video foundation remains external. A sequence is considered
visually connected only when every accepted render carries continuity provenance
bound to the exact current artifact and, for continuation shots, to the exact
predecessor render artifact and request hash.

The validator is intentionally fail-closed. Ordered MP4 files, matching shot IDs,
or prompt-level ``previous_shot`` metadata are not sufficient production evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .gpu_foundation_smoke import GPUFoundationExecutionReceipt
from .production_continuity_diffusers import VISUAL_CONTINUITY_SCHEMA


class ConnectedContinuityEvidenceError(RuntimeError):
    """Raised when connected render evidence is missing, stale, or substituted."""


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ConnectedContinuityEvidenceError(f"{field} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ConnectedContinuityEvidenceError(
            f"{field} must be a SHA-256 hex digest"
        ) from exc
    return value.lower()


def _provenance(receipt: GPUFoundationExecutionReceipt) -> Mapping[str, Any]:
    provenance = getattr(receipt.result, "conditioning_provenance", None)
    if not isinstance(provenance, Mapping):
        raise ConnectedContinuityEvidenceError(
            "connected production evidence requires returned conditioning provenance"
        )
    if provenance.get("schema") != VISUAL_CONTINUITY_SCHEMA:
        raise ConnectedContinuityEvidenceError(
            "connected production evidence has unsupported continuity schema"
        )
    return provenance


def validate_connected_visual_continuity(
    receipts: Sequence[GPUFoundationExecutionReceipt],
) -> tuple[dict[str, Any], ...]:
    """Validate exact artifact lineage across a rendered connected sequence.

    The first receipt must be a root render. Every later receipt must explicitly
    name the immediately preceding scene/shot and cryptographically bind its visual
    handoff to that predecessor's accepted artifact and request hash. Current-shot
    bindings are checked against the receipt rather than trusted from provenance.
    """

    if not receipts:
        raise ConnectedContinuityEvidenceError(
            "connected production evidence requires at least one render receipt"
        )

    normalized: list[dict[str, Any]] = []
    previous: GPUFoundationExecutionReceipt | None = None

    for index, receipt in enumerate(receipts):
        result = receipt.result
        provenance = _provenance(receipt)

        if provenance.get("scene_id") != result.scene_id:
            raise ConnectedContinuityEvidenceError(
                "continuity provenance scene id does not match rendered result"
            )
        if provenance.get("shot_id") != result.shot_id:
            raise ConnectedContinuityEvidenceError(
                "continuity provenance shot id does not match rendered result"
            )

        current_artifact = _require_sha256(
            provenance.get("current_artifact_sha256"),
            field="current_artifact_sha256",
        )
        receipt_artifact = _require_sha256(
            receipt.output_sha256,
            field="receipt.output_sha256",
        )
        if current_artifact != receipt_artifact:
            raise ConnectedContinuityEvidenceError(
                "continuity provenance current artifact does not match render receipt"
            )
        if provenance.get("current_request_hash") != result.request_hash:
            raise ConnectedContinuityEvidenceError(
                "continuity provenance current request hash does not match render result"
            )

        if previous is None:
            if provenance.get("mode") not in {
                "approved_reference_root",
                "text_only_root",
            }:
                raise ConnectedContinuityEvidenceError(
                    "first connected render must carry root continuity provenance"
                )
            if provenance.get("previous_scene_id") is not None:
                raise ConnectedContinuityEvidenceError(
                    "root continuity provenance must not name a previous scene"
                )
            if provenance.get("previous_shot_id") is not None:
                raise ConnectedContinuityEvidenceError(
                    "root continuity provenance must not name a previous shot"
                )
            if provenance.get("predecessor_artifact_sha256") is not None:
                raise ConnectedContinuityEvidenceError(
                    "root continuity provenance must not bind a predecessor artifact"
                )
            if provenance.get("predecessor_request_hash") is not None:
                raise ConnectedContinuityEvidenceError(
                    "root continuity provenance must not bind a predecessor request"
                )
            if provenance.get("in_memory_terminal_frame") is not False:
                raise ConnectedContinuityEvidenceError(
                    "root continuity provenance must not claim terminal-frame handoff"
                )
        else:
            previous_result = previous.result
            if provenance.get("mode") != "predecessor_terminal_frame_lineage":
                raise ConnectedContinuityEvidenceError(
                    "continuation render lacks predecessor terminal-frame lineage"
                )
            if provenance.get("previous_scene_id") != previous_result.scene_id:
                raise ConnectedContinuityEvidenceError(
                    "continuation provenance does not name the immediate predecessor scene"
                )
            if provenance.get("previous_shot_id") != previous_result.shot_id:
                raise ConnectedContinuityEvidenceError(
                    "continuation provenance does not name the immediate predecessor shot"
                )
            predecessor_artifact = _require_sha256(
                provenance.get("predecessor_artifact_sha256"),
                field="predecessor_artifact_sha256",
            )
            previous_artifact = _require_sha256(
                previous.output_sha256,
                field="previous.output_sha256",
            )
            if predecessor_artifact != previous_artifact:
                raise ConnectedContinuityEvidenceError(
                    "continuation provenance is bound to a different predecessor artifact"
                )
            if (
                provenance.get("predecessor_request_hash")
                != previous_result.request_hash
            ):
                raise ConnectedContinuityEvidenceError(
                    "continuation provenance is bound to a different predecessor request"
                )
            if provenance.get("in_memory_terminal_frame") is not True:
                raise ConnectedContinuityEvidenceError(
                    "continuation provenance does not attest terminal-frame handoff"
                )

        normalized.append(dict(provenance))
        previous = receipt

    return tuple(normalized)


def production_visual_continuity_evidence(
    receipts: Sequence[GPUFoundationExecutionReceipt],
) -> bool:
    """Return whether receipts constitute artifact-bound visual continuity evidence."""

    try:
        validate_connected_visual_continuity(receipts)
    except ConnectedContinuityEvidenceError:
        return False
    return True


__all__ = [
    "ConnectedContinuityEvidenceError",
    "production_visual_continuity_evidence",
    "validate_connected_visual_continuity",
]
