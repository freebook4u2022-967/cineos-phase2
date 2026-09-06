import json

import pytest

from cineos.atlas import gpu_benchmark_cli as cli
from cineos.atlas.gpu_benchmark_cli import (
    GPUProductionBenchmarkCLIError,
    load_native_requests,
    run_production_benchmark,
)
from cineos.atlas.native_request import NativeShotRequest


def _request(index: int) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-shot-count",
        camera={"movement": "tracking"},
        characters=[{"character_id": "lead"}, {"character_id": "partner"}],
        environment={"location": "street"},
        wardrobe=[],
        props=[{"prop_id": "case"}],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={
            "action": "walk",
            "dialogue_timing": [
                {"speaker_id": "lead", "start_seconds": 0.2, "end_seconds": 1.0}
            ],
        },
        approved_reference_ids=[
            "lead-approved-reference",
            "partner-approved-reference",
        ],
        deterministic_seed=5000 + index,
        renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
        metadata={
            cli.COMPETITIVE_CHALLENGE_METADATA_KEY: sorted(
                cli.REQUIRED_COMPETITIVE_CHALLENGES
            )
        },
    )
    request.refresh_hash()
    return request


def _write_manifest(tmp_path, shot_count: int):
    source = tmp_path / f"requests-{shot_count}.json"
    source.write_text(
        json.dumps(
            {"shots": [_request(index).to_dict() for index in range(shot_count)]}
        ),
        encoding="utf-8",
    )
    return source


@pytest.mark.parametrize("shot_count", [4, 11])
def test_load_native_requests_rejects_out_of_range_connected_film(tmp_path, shot_count):
    source = _write_manifest(tmp_path, shot_count)

    with pytest.raises(
        GPUProductionBenchmarkCLIError,
        match=rf"requires 5-10 shots; received {shot_count}",
    ):
        load_native_requests(source)


@pytest.mark.parametrize("shot_count", [5, 10])
def test_load_native_requests_accepts_connected_film_boundaries(tmp_path, shot_count):
    source = _write_manifest(tmp_path, shot_count)

    loaded = load_native_requests(source)

    assert len(loaded) == shot_count


@pytest.mark.parametrize("shot_count", [0, 1, 4, 11])
def test_direct_production_runner_rejects_invalid_count_before_qc_model_load(
    monkeypatch, tmp_path, shot_count
):
    qc_loaded = False

    def unexpected_quality_evaluator(*args, **kwargs):
        nonlocal qc_loaded
        qc_loaded = True
        return object()

    monkeypatch.setattr(
        cli, "_production_quality_evaluator", unexpected_quality_evaluator
    )

    with pytest.raises(
        GPUProductionBenchmarkCLIError,
        match=rf"requires 5-10 shots; received {shot_count}",
    ):
        run_production_benchmark(
            "production-evidence",
            [_request(index) for index in range(shot_count)],
            output_dir=tmp_path / "renders",
            reference_manifest="approved-references.json",
        )

    assert qc_loaded is False
