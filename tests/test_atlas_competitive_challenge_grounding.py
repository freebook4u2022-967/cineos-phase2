import pytest

from cineos.atlas import gpu_benchmark_cli as cli
from cineos.atlas.gpu_benchmark_cli import GPUProductionBenchmarkCLIError
from cineos.atlas.native_request import NativeShotRequest


def _request(index: int) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-motion-grounding",
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
        approved_reference_ids=["lead-ref", "partner-ref"],
        deterministic_seed=7000 + index,
        renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
        metadata={
            cli.COMPETITIVE_CHALLENGE_METADATA_KEY: sorted(
                cli.REQUIRED_COMPETITIVE_CHALLENGES
            )
        },
    )
    request.refresh_hash()
    return request


def _requests() -> list[NativeShotRequest]:
    return [_request(index) for index in range(5)]


def test_connected_challenge_grounding_accepts_explicit_walk_and_tracking_motion():
    cli._validate_connected_sequence(_requests())


def test_walking_running_challenge_rejects_non_locomotion_action():
    requests = _requests()
    requests[0].performance["action"] = "stand and talk"
    requests[0].refresh_hash()

    with pytest.raises(GPUProductionBenchmarkCLIError, match="no explicit walk/run"):
        cli._validate_connected_sequence(requests)


def test_walking_running_challenge_accepts_native_body_performance_track():
    requests = _requests()
    requests[0].performance["action"] = "dialogue"
    requests[0].performance["body_performance_tracks"] = [
        {"character_id": "lead", "action": "sprinting"}
    ]
    requests[0].refresh_hash()

    cli._validate_connected_sequence(requests)


def test_fast_camera_challenge_rejects_static_camera_conditioning():
    requests = _requests()
    requests[0].camera["movement"] = "locked-off"
    requests[0].refresh_hash()

    with pytest.raises(GPUProductionBenchmarkCLIError, match="non-static camera"):
        cli._validate_connected_sequence(requests)
