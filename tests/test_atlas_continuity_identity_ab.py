import copy

import pytest

from cineos.atlas.continuity_identity_ab import (
    ContinuityIdentityABError,
    evaluate_continuity_identity_ab,
)


def _receipt(
    *,
    chain: str,
    identity: float,
    temporal: float = 0.94,
    motion: float = 0.92,
    artifact: float = 1.0,
):
    reports = []
    for index in range(5):
        reports.append(
            {
                "accepted": True,
                "scene_id": "scene-ab",
                "shot_id": f"shot-{index}",
                "measurement": {
                    "metrics": {
                        "identity_similarity": identity,
                        "temporal_consistency": temporal,
                        "motion_quality": motion,
                        "artifact_integrity": artifact,
                    }
                },
            }
        )
    return {
        "schema": "cineos-gpu-connected-benchmark/0.2",
        "profile_id": "wan22-ti2v-5b",
        "chain_sha256": chain * 64,
        "production_gpu_evidence": True,
        "production_quality_evidence": True,
        "evidence_tier": "production-gpu-quality-gated",
        "quality_reports": reports,
    }


def test_ab_gate_promotes_measured_identity_gain_without_quality_regression():
    baseline = _receipt(chain="a", identity=0.80)
    candidate = _receipt(chain="b", identity=0.84, temporal=0.935, motion=0.915)

    decision = evaluate_continuity_identity_ab(baseline, candidate)

    assert decision.promotable is True
    assert decision.shot_count == 5
    assert decision.deltas["identity_similarity"] == pytest.approx(0.04)
    assert decision.failed_criteria == ()
    assert decision.to_dict()["schema"] == "cineos-continuity-identity-ab-decision/0.1"


def test_ab_gate_rejects_candidate_with_temporal_regression():
    baseline = _receipt(chain="a", identity=0.80, temporal=0.94)
    candidate = _receipt(chain="b", identity=0.85, temporal=0.90)

    decision = evaluate_continuity_identity_ab(baseline, candidate)

    assert decision.promotable is False
    assert "temporal_regression" in decision.failed_criteria


def test_ab_gate_rejects_hidden_per_shot_identity_regression():
    baseline = _receipt(chain="a", identity=0.80)
    candidate = _receipt(chain="b", identity=0.84)
    candidate["quality_reports"][2]["measurement"]["metrics"][
        "identity_similarity"
    ] = 0.70
    candidate["quality_reports"][0]["measurement"]["metrics"][
        "identity_similarity"
    ] = 1.0
    candidate["quality_reports"][1]["measurement"]["metrics"][
        "identity_similarity"
    ] = 1.0

    decision = evaluate_continuity_identity_ab(baseline, candidate)

    assert decision.promotable is False
    assert "per_shot_identity_regression" in decision.failed_criteria


def test_ab_gate_requires_production_quality_evidence():
    baseline = _receipt(chain="a", identity=0.80)
    candidate = _receipt(chain="b", identity=0.84)
    candidate["production_quality_evidence"] = False

    with pytest.raises(ContinuityIdentityABError, match="production measured QC"):
        evaluate_continuity_identity_ab(baseline, candidate)


def test_ab_gate_rejects_reused_render_chain():
    baseline = _receipt(chain="a", identity=0.80)
    candidate = copy.deepcopy(baseline)
    candidate["quality_reports"][0]["measurement"]["metrics"][
        "identity_similarity"
    ] = 0.90

    with pytest.raises(ContinuityIdentityABError, match="same rendered chain"):
        evaluate_continuity_identity_ab(baseline, candidate)


def test_ab_gate_requires_same_ordered_shots():
    baseline = _receipt(chain="a", identity=0.80)
    candidate = _receipt(chain="b", identity=0.84)
    candidate["quality_reports"][1]["shot_id"] = "different-shot"

    with pytest.raises(ContinuityIdentityABError, match="same ordered shots"):
        evaluate_continuity_identity_ab(baseline, candidate)
