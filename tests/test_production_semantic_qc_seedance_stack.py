from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from cineos.atlas.latentsync_syncnet_scorer import LatentSyncSyncNetScorer
from cineos.atlas.production_semantic_qc import (
    build_seedance_challenge_semantic_scorer,
)
from cineos.atlas.qwen25vl_semantic_judge import (
    QWEN25VL_METRICS,
    Qwen25VLSemanticJudge,
)


class _CoreScorer:
    semantic_measurement_evidence = True

    def __call__(self, *args, **kwargs):
        return {"identity_similarity": 0.9, "motion_quality": 0.9}

    def runtime_provenance(self):
        return {
            "schema": "test-core/0.1",
            "origin": "external_pretrained",
            "production_measurement_evidence": True,
        }


def _av_scorer(tmp_path: Path) -> LatentSyncSyncNetScorer:
    checkpoint = tmp_path / "syncnet.model"
    checkpoint.write_bytes(b"checkpoint")
    return LatentSyncSyncNetScorer(
        repository_root=tmp_path / "LatentSync",
        checkpoint_path=checkpoint,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )


def test_seedance_stack_has_disjoint_visual_and_av_metric_ownership(tmp_path: Path) -> None:
    shots = [
        SimpleNamespace(
            metadata={
                "competitive_challenges": [
                    "multi_character_interaction",
                    "hands_anatomy",
                    "walking_running",
                    "dialogue_lip_sync",
                    "object_interaction",
                    "fast_camera_movement",
                    "lighting_changes",
                    "physics",
                ]
            }
        )
    ]
    visual = Qwen25VLSemanticJudge(model=object(), processor=object())
    ensemble = build_seedance_challenge_semantic_scorer(
        _CoreScorer(),
        shots,
        visual_judge=visual,
        av_sync_scorer=_av_scorer(tmp_path),
    )

    assert ensemble.metric_owners["dialogue_lip_sync"] == "latentsync_syncnet_av"
    for metric in QWEN25VL_METRICS:
        assert ensemble.metric_owners[metric] == "qwen25vl_visual_difficult_cases"
    assert "dialogue_lip_sync" not in QWEN25VL_METRICS
    assert ensemble.semantic_measurement_evidence is True

    provenance = ensemble.runtime_provenance()
    components = {item["name"]: item for item in provenance["components"]}
    assert components["qwen25vl_visual_difficult_cases"]["scorer"]["origin"] == (
        "external_pretrained"
    )
    assert components["latentsync_syncnet_av"]["scorer"]["origin"] == (
        "external_pretrained"
    )
