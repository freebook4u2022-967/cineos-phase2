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
        environment={"location": "street", "lighting": "day_to_night transition"},
        wardrobe=[],
        props=[{"prop_id": "case", "action": "throwing"}],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={
            "action": "walk while throwing case",
            "gesture_tracks": [
                {"character_id": "lead", "action": "gripping with both hands"}
            ],
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


def test_connected_challenge_grounding_accepts_explicit_difficult_conditioning():
    cli._validate_connected_sequence(_requests())


def test_walking_running_challenge_rejects_non_locomotion_action():
    requests = _requests()
    requests[0].performance["action"] = "stand and throw case"
    requests[0].refresh_hash()

    with pytest.raises(GPUProductionBenchmarkCLIError, match="no explicit walk/run"):
        cli._validate_connected_sequence(requests)


def test_walking_running_challenge_accepts_native_body_performance_track():
    requests = _requests()
    requests[0].performance["action"] = "throw case"
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


def test_hands_anatomy_challenge_rejects_missing_hand_or_gesture_conditioning():
    requests = _requests()
    requests[0].performance["gesture_tracks"] = []
    requests[0].performance["action"] = "walk while throwing case"
    requests[0].props = [{"prop_id": "case", "action": "throwing"}]
    requests[0].refresh_hash()

    with pytest.raises(GPUProductionBenchmarkCLIError, match="hand/gesture"):
        cli._validate_connected_sequence(requests)


def test_hands_anatomy_challenge_accepts_explicit_reaching_action():
    requests = _requests()
    requests[0].performance["gesture_tracks"] = []
    requests[0].performance["action"] = "walk reach throw case"
    requests[0].refresh_hash()

    cli._validate_connected_sequence(requests)


def test_lighting_changes_challenge_rejects_static_lighting_description():
    requests = _requests()
    requests[0].environment = {"location": "street", "lighting": "daylight"}
    requests[0].refresh_hash()

    with pytest.raises(GPUProductionBenchmarkCLIError, match="lighting-transition"):
        cli._validate_connected_sequence(requests)


def test_lighting_changes_challenge_accepts_metadata_transition():
    requests = _requests()
    requests[0].environment = {"location": "street", "lighting": "daylight"}
    requests[0].metadata["lighting_transition"] = "sunset"
    requests[0].refresh_hash()

    cli._validate_connected_sequence(requests)


def test_physics_challenge_rejects_static_prop_presence():
    requests = _requests()
    requests[0].performance["action"] = "walk while holding case"
    requests[0].props = [{"prop_id": "case"}]
    requests[0].refresh_hash()

    with pytest.raises(GPUProductionBenchmarkCLIError, match="physical-interaction"):
        cli._validate_connected_sequence(requests)


def test_physics_challenge_accepts_explicit_dynamic_prop_action():
    requests = _requests()
    requests[0].performance["action"] = "walk while holding case"
    requests[0].props = [{"prop_id": "case", "action": "dropping"}]
    requests[0].refresh_hash()

    cli._validate_connected_sequence(requests)


def test_dialogue_lip_sync_requires_speaker_identity_grounding():
    request = _request(0)
    request.performance["dialogue_timing"] = [
        {"start_seconds": 0.2, "end_seconds": 1.0}
    ]

    with pytest.raises(ValueError, match="requires speaker_id"):
        request.validate_timing_integrity()


def test_dialogue_lip_sync_rejects_unknown_speaker_identity():
    request = _request(0)
    request.performance["dialogue_timing"] = [
        {"speaker_id": "intruder", "start_seconds": 0.2, "end_seconds": 1.0}
    ]

    with pytest.raises(ValueError, match="conditioned character_id"):
        request.validate_timing_integrity()


def test_dialogue_timing_legacy_without_competitive_tag_remains_compatible():
    request = _request(0)
    request.metadata.pop(cli.COMPETITIVE_CHALLENGE_METADATA_KEY)
    request.performance["dialogue_timing"] = [
        {"start_seconds": 0.2, "end_seconds": 1.0}
    ]

    request.validate_timing_integrity()
