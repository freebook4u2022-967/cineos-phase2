import pytest

from cineos.atlas import gpu_benchmark_cli as cli
from cineos.atlas.gpu_benchmark_cli import GPUProductionBenchmarkCLIError
from cineos.atlas.native_request import NativeShotRequest


def _request(index: int) -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id=f"shot-{index}",
        scene_id="scene-motion-grounding",
        camera={"movement": "whip_pan"},
        characters=[{"character_id": "lead"}, {"character_id": "partner"}],
        environment={"location": "street", "lighting": "day_to_night transition"},
        wardrobe=[],
        props=[{"prop_id": "case", "action": "throwing"}],
        continuity={"previous_shot": None if index == 0 else f"shot-{index - 1}"},
        performance={
            "action": "walk while throwing case",
            "interaction_cues": [
                {
                    "participant_ids": ["lead", "partner"],
                    "action": "lead hands the case to partner",
                }
            ],
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


def test_multi_character_interaction_rejects_presence_without_interaction_cue():
    requests = _requests()
    requests[0].performance.pop("interaction_cues")
    requests[0].refresh_hash()

    with pytest.raises(GPUProductionBenchmarkCLIError, match="interaction_cues"):
        cli._validate_connected_sequence(requests)


def test_multi_character_interaction_rejects_single_participant_cue():
    requests = _requests()
    requests[0].performance["interaction_cues"] = [
        {"participant_ids": ["lead"], "action": "lead turns"}
    ]
    requests[0].refresh_hash()

    with pytest.raises(GPUProductionBenchmarkCLIError, match="at least two distinct"):
        cli._validate_connected_sequence(requests)


def test_multi_character_interaction_rejects_unconditioned_participant():
    requests = _requests()
    requests[0].performance["interaction_cues"] = [
        {
            "participant_ids": ["lead", "intruder"],
            "action": "lead hands the case to intruder",
        }
    ]
    requests[0].refresh_hash()

    with pytest.raises(GPUProductionBenchmarkCLIError, match="unconditioned"):
        cli._validate_connected_sequence(requests)


def test_multi_character_interaction_accepts_grounded_two_character_cue():
    requests = _requests()
    requests[0].performance["interaction_cues"] = [
        {
            "participant_ids": ["lead", "partner"],
            "description": "lead and partner exchange the case while crossing paths",
        }
    ]
    requests[0].refresh_hash()

    cli._validate_connected_sequence(requests)


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

    with pytest.raises(GPUProductionBenchmarkCLIError, match="fast/aggressive camera"):
        cli._validate_connected_sequence(requests)


def test_fast_camera_challenge_rejects_ordinary_tracking_motion():
    requests = _requests()
    requests[0].camera["movement"] = "tracking"
    requests[0].refresh_hash()

    with pytest.raises(GPUProductionBenchmarkCLIError, match="fast/aggressive camera"):
        cli._validate_connected_sequence(requests)


def test_fast_camera_challenge_accepts_explicit_speed_cue():
    requests = _requests()
    requests[0].camera["movement"] = {"type": "tracking", "speed": "fast"}
    requests[0].refresh_hash()

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

    with pytest.raises(ValueError, match="conditioned character identity"):
        request.validate_timing_integrity()


def test_dialogue_lip_sync_accepts_canonical_character_uuid():
    request = _request(0)
    request.characters = [
        {"character_uuid": "lead", "approved_reference_ids": ["lead-ref"]},
        {"character_uuid": "partner", "approved_reference_ids": ["partner-ref"]},
    ]
    request.refresh_hash()

    request.validate_timing_integrity()


def test_dialogue_lip_sync_rejects_conflicting_character_identity_aliases():
    request = _request(0)
    request.characters = [
        {"character_uuid": "lead", "character_id": "different-lead"},
        {"character_uuid": "partner"},
    ]

    with pytest.raises(ValueError, match="conflicting character_uuid/character_id"):
        request.validate_timing_integrity()


def test_dialogue_lip_sync_rejects_duplicate_canonical_character_uuid():
    request = _request(0)
    request.characters = [
        {"character_uuid": "lead"},
        {"character_uuid": "lead"},
    ]

    with pytest.raises(ValueError, match="unique conditioned character identities"):
        request.validate_timing_integrity()


def test_dialogue_lip_sync_rejects_duplicate_identity_across_aliases():
    request = _request(0)
    request.characters = [
        {"character_uuid": "lead"},
        {"character_id": "lead"},
    ]

    with pytest.raises(ValueError, match="duplicate 'lead'"):
        request.validate_timing_integrity()


def test_dialogue_timing_legacy_without_competitive_tag_remains_compatible():
    request = _request(0)
    request.metadata.pop(cli.COMPETITIVE_CHALLENGE_METADATA_KEY)
    request.performance["dialogue_timing"] = [
        {"start_seconds": 0.2, "end_seconds": 1.0}
    ]

    request.validate_timing_integrity()


def test_dialogue_timing_rejects_camera_duration_conflicting_with_renderer_contract():
    request = _request(0)
    request.camera["duration"] = 1.5

    with pytest.raises(ValueError, match="camera.duration conflicts"):
        request.validate_timing_integrity()


def test_frame_dialogue_rejects_camera_fps_conflicting_with_renderer_contract():
    request = _request(0)
    request.camera["fps"] = 30.0
    request.performance["dialogue_timing"] = [
        {"speaker_id": "lead", "start_frame": 6, "end_frame": 24}
    ]

    with pytest.raises(ValueError, match="camera.fps conflicts"):
        request.validate_timing_integrity()


def test_camera_only_duration_bounds_dialogue_timing():
    request = _request(0)
    request.renderer_requirements.pop("duration_seconds")
    request.camera["duration"] = 0.75

    with pytest.raises(ValueError, match="extends beyond the shot duration"):
        request.validate_timing_integrity()
