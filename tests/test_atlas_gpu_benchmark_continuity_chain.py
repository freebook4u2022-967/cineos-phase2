import json

import pytest

from cineos.atlas import gpu_benchmark_cli as cli
from cineos.atlas.gpu_benchmark_cli import (
    GPUProductionBenchmarkCLIError,
    load_native_requests,
    run_production_benchmark,
)
from cineos.atlas.native_request import NativeShotRequest


def _request(index: int, *, predecessor=...):
    if predecessor is ...:
        predecessor = None if index == 0 else f"shot-{index - 1}"
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-continuity",
        camera={"movement": "whip_pan"},
        characters=[{"character_id": "lead"}, {"character_id": "partner"}],
        environment={"location": "street", "lighting": "day_to_night transition"},
        wardrobe=[],
        props=[{"prop_id": "case", "action": "throwing"}],
        continuity={"previous_shot_id": predecessor},
        performance={
            "action": "walk while throwing case",
            "gesture_tracks": [
                {"character_id": "lead", "action": "gripping with both hands"}
            ],
            "interaction_cues": [
                {
                    "participant_ids": ["lead", "partner"],
                    "action": "lead hands the case to partner",
                }
            ],
            "object_interaction_cues": [
                {
                    "character_id": "lead",
                    "prop_id": "case",
                    "action": "lead handing the case to partner",
                }
            ],
            "dialogue_timing": [
                {"speaker_id": "lead", "start_seconds": 0.2, "end_seconds": 1.0}
            ],
        },
        approved_reference_ids=[
            "lead-approved-reference",
            "partner-approved-reference",
        ],
        deterministic_seed=6000 + index,
        renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
        metadata={
            cli.COMPETITIVE_CHALLENGE_METADATA_KEY: sorted(
                cli.REQUIRED_COMPETITIVE_CHALLENGES
            )
        },
    )
    request.refresh_hash()
    return request


def _write_manifest(tmp_path, requests):
    source = tmp_path / "requests.json"
    source.write_text(
        json.dumps({"shots": [request.to_dict() for request in requests]}),
        encoding="utf-8",
    )
    return source


def test_manifest_rejects_disconnected_predecessor_chain(tmp_path):
    requests = [_request(index) for index in range(5)]
    requests[3] = _request(3, predecessor="shot-0")

    with pytest.raises(
        GPUProductionBenchmarkCLIError,
        match="continuity is not a contiguous ordered chain",
    ):
        load_native_requests(_write_manifest(tmp_path, requests))


def test_manifest_rejects_first_shot_with_predecessor(tmp_path):
    requests = [_request(index) for index in range(5)]
    requests[0] = _request(0, predecessor="external-shot")

    with pytest.raises(
        GPUProductionBenchmarkCLIError,
        match="first shot.*must not declare a predecessor",
    ):
        load_native_requests(_write_manifest(tmp_path, requests))


def test_manifest_rejects_duplicate_shot_ids(tmp_path):
    requests = [_request(index) for index in range(5)]
    requests[4].shot_id = "shot-3"
    requests[4].refresh_hash()

    with pytest.raises(
        GPUProductionBenchmarkCLIError,
        match="requires unique shot_id values",
    ):
        load_native_requests(_write_manifest(tmp_path, requests))


def test_manifest_accepts_canonical_connected_predecessor_chain(tmp_path):
    requests = [_request(index) for index in range(5)]

    loaded = load_native_requests(_write_manifest(tmp_path, requests))

    assert [request.shot_id for request in loaded] == [
        "shot-0",
        "shot-1",
        "shot-2",
        "shot-3",
        "shot-4",
    ]


def test_direct_runner_rejects_disconnected_chain_before_qc_model_load(
    monkeypatch, tmp_path
):
    evaluator_loaded = False

    def unexpected_quality_evaluator(*args, **kwargs):
        nonlocal evaluator_loaded
        evaluator_loaded = True
        return object()

    monkeypatch.setattr(cli, "_production_quality_evaluator", unexpected_quality_evaluator)
    requests = [_request(index) for index in range(5)]
    requests[2] = _request(2, predecessor="shot-0")

    with pytest.raises(
        GPUProductionBenchmarkCLIError,
        match="continuity is not a contiguous ordered chain",
    ):
        run_production_benchmark(
            "continuity-chain",
            requests,
            output_dir=tmp_path,
            reference_manifest="references.json",
        )

    assert evaluator_loaded is False
