"""Integration regression for the quality-first A14B resident execution boundary."""

import pytest

from cineos.atlas import quality_first_gpu_benchmark_cli as cli
from cineos.atlas.gpu_benchmark_cli import GPUProductionBenchmarkCLIError
from cineos.atlas.gpu_preflight import GPUDeviceProfile
from cineos.atlas.native_request import NativeShotRequest


def _request(index: int) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="a14b-resident-boundary",
        camera={
            "resolution": [832, 480],
            "fps": 16.0,
            "duration": 2.0,
            "movement": "tracking",
        },
        characters=[{"character_id": "hero"}],
        environment={"lighting": "daylight"},
        wardrobe=[],
        props=[],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={"action": "walking"},
        approved_reference_ids=["hero-reference"],
        deterministic_seed=9100 + index,
        renderer_requirements={
            "supported_resolution": [832, 480],
            "supported_fps": 16.0,
            "maximum_duration": 2.0,
        },
    )
    request.refresh_hash()
    return request


def test_entrypoint_rejects_unvalidated_a14b_offload_before_renderer(
    monkeypatch, tmp_path
):
    renderer_called = False

    def unexpected_render(*args, **kwargs):
        nonlocal renderer_called
        renderer_called = True
        raise AssertionError(
            "renderer must not run for an unvalidated A14B offload plan"
        )

    monkeypatch.setattr(
        cli,
        "run_production_quality_retry_connected_gpu_benchmark",
        unexpected_render,
    )
    device = GPUDeviceProfile(
        index=0,
        name="test-gpu",
        compute_capability=(9, 0),
        total_vram_gb=96.0,
        free_vram_gb=90.0,
        supports_bfloat16=True,
    )

    with pytest.raises(
        GPUProductionBenchmarkCLIError,
        match="requires validated resident execution; generic planner selected model_cpu_offload",
    ):
        cli.run_quality_first_production_benchmark(
            "a14b-resident-boundary",
            [_request(index) for index in range(5)],
            output_dir=tmp_path,
            reference_manifest="references.json",
            devices=(device,),
        )

    assert renderer_called is False
