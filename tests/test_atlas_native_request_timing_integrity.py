import math

import pytest

from cineos.atlas.native_request import NativeShotRequest


def _request() -> NativeShotRequest:
    return NativeShotRequest(
        shot_id="shot-01",
        scene_id="scene-01",
        camera={"movement": "tracking"},
        characters=[{"character_id": "lead"}],
        environment={"location": "street"},
        wardrobe=[],
        props=[],
        continuity={"previous_shot": None},
        performance={
            "dialogue_timing": [
                {"speaker_id": "lead", "start_seconds": 0.25, "end_seconds": 1.25}
            ]
        },
        approved_reference_ids=["lead-ref"],
        deterministic_seed=1234,
        renderer_requirements={"fps": 24.0, "duration_seconds": 2.0},
    )


def test_native_request_accepts_finite_in_bounds_dialogue_timing():
    request = _request()

    digest = request.refresh_hash()

    assert len(digest) == 64
    assert request.content_hash == digest


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf, True])
def test_native_request_rejects_invalid_fps(value):
    request = _request()
    request.renderer_requirements["fps"] = value

    with pytest.raises(ValueError, match="renderer_requirements.fps"):
        request.refresh_hash()


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, -math.inf, False])
def test_native_request_rejects_invalid_duration(value):
    request = _request()
    request.renderer_requirements["duration_seconds"] = value

    with pytest.raises(ValueError, match="renderer_requirements.duration_seconds"):
        request.refresh_hash()


def test_native_request_rejects_dialogue_that_extends_beyond_shot_duration():
    request = _request()
    request.performance["dialogue_timing"][0]["end_seconds"] = 2.01

    with pytest.raises(ValueError, match="extends beyond the shot duration"):
        request.refresh_hash()


@pytest.mark.parametrize(
    ("start_seconds", "end_seconds"),
    [
        (-0.1, 1.0),
        (math.nan, 1.0),
        (0.1, math.inf),
        (1.0, 1.0),
        (1.1, 1.0),
    ],
)
def test_native_request_rejects_invalid_dialogue_interval(start_seconds, end_seconds):
    request = _request()
    cue = request.performance["dialogue_timing"][0]
    cue["start_seconds"] = start_seconds
    cue["end_seconds"] = end_seconds

    with pytest.raises(ValueError, match="dialogue_timing"):
        request.refresh_hash()


def test_native_request_keeps_legacy_omitted_timing_fields_compatible():
    request = _request()
    request.performance = {}
    request.renderer_requirements = {}

    digest = request.refresh_hash()

    assert len(digest) == 64
