from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from cineos.atlas.production_retry_evidence import (
    ProductionRetryEvidenceError,
    validate_production_quality_retry_gate,
)


def _digest(char: str) -> str:
    return char * 64


def _receipt(*, scene_id: str, shot_id: str, request_hash: str, output_hash: str):
    return SimpleNamespace(
        result=SimpleNamespace(
            scene_id=scene_id,
            shot_id=shot_id,
            request_hash=request_hash,
        ),
        output_sha256=output_hash,
    )


def _valid_gate():
    first_request = _digest("1")
    first_output = _digest("a")
    second_original = _digest("2")
    second_retry = _digest("3")
    second_rejected_output = _digest("b")
    second_output = _digest("c")
    transition = {"attempt_index": 1, "accepted": True}
    gate = {
        "schema": "cineos-gpu-quality-retry-gate/0.2",
        "accepted": True,
        "policy": {"max_attempts": 3, "seed_stride": 104729},
        "shot_count": 2,
        "shots": [
            {
                "scene_id": "scene-1",
                "shot_id": "shot-1",
                "original_request_hash": first_request,
                "accepted_request_hash": first_request,
                "attempt_count": 1,
                "attempts": [
                    {
                        "attempt_index": 0,
                        "original_request_hash": first_request,
                        "effective_request_hash": first_request,
                        "seed": 10,
                        "output_sha256": first_output,
                        "accepted": True,
                    }
                ],
                "transition_attempts": [],
                "accepted_transition": None,
            },
            {
                "scene_id": "scene-1",
                "shot_id": "shot-2",
                "original_request_hash": second_original,
                "accepted_request_hash": second_retry,
                "attempt_count": 2,
                "attempts": [
                    {
                        "attempt_index": 0,
                        "original_request_hash": second_original,
                        "effective_request_hash": second_original,
                        "seed": 20,
                        "output_sha256": second_rejected_output,
                        "accepted": False,
                    },
                    {
                        "attempt_index": 1,
                        "original_request_hash": second_original,
                        "effective_request_hash": second_retry,
                        "seed": 104749,
                        "output_sha256": second_output,
                        "accepted": True,
                    },
                ],
                "transition_attempts": [transition],
                "accepted_transition": transition,
            },
        ],
        "transition_gate_applied": True,
        "accepted_transition_count": 1,
        "accepted_transitions": [transition],
    }
    receipts = (
        _receipt(
            scene_id="scene-1",
            shot_id="shot-1",
            request_hash=first_request,
            output_hash=first_output,
        ),
        _receipt(
            scene_id="scene-1",
            shot_id="shot-2",
            request_hash=second_retry,
            output_hash=second_output,
        ),
    )
    return gate, receipts


def test_validates_recovered_connected_shot_lineage():
    gate, receipts = _valid_gate()

    evidence = validate_production_quality_retry_gate(gate, receipts)

    assert evidence == {
        "schema": "cineos-production-retry-lineage-validation/0.1",
        "verified": True,
        "shot_count": 2,
        "total_attempts": 3,
        "rejected_attempts": 1,
        "recovered_shot_ids": ["shot-2"],
        "transition_gate_applied": True,
    }


def test_rejects_accepted_attempt_before_final_attempt():
    gate, receipts = _valid_gate()
    gate = deepcopy(gate)
    gate["shots"][1]["attempts"][0]["accepted"] = True

    with pytest.raises(
        ProductionRetryEvidenceError,
        match="accepted attempt before the final attempt",
    ):
        validate_production_quality_retry_gate(gate, receipts)


def test_rejects_non_deterministic_retry_seed_progression():
    gate, receipts = _valid_gate()
    gate = deepcopy(gate)
    gate["shots"][1]["attempts"][1]["seed"] += 1

    with pytest.raises(ProductionRetryEvidenceError, match="seed progression"):
        validate_production_quality_retry_gate(gate, receipts)


def test_rejects_final_artifact_substitution():
    gate, receipts = _valid_gate()
    gate = deepcopy(gate)
    gate["shots"][1]["attempts"][1]["output_sha256"] = _digest("d")

    with pytest.raises(ProductionRetryEvidenceError, match="accepted GPU artifact"):
        validate_production_quality_retry_gate(gate, receipts)


def test_allows_shot_quality_pass_followed_by_transition_rejection():
    gate, receipts = _valid_gate()
    gate = deepcopy(gate)
    first_attempt = gate["shots"][1]["attempts"][0]
    first_attempt["accepted"] = True
    gate["shots"][1]["transition_attempts"].insert(
        0, {"attempt_index": 0, "accepted": False}
    )

    evidence = validate_production_quality_retry_gate(gate, receipts)

    assert evidence["rejected_attempts"] == 1
    assert evidence["recovered_shot_ids"] == ["shot-2"]
