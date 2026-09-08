from cineos.atlas.native_request import NativeShotRequest
from cineos.atlas.production_diffusers import ProductionDiffusersVideoRenderer


def _request() -> NativeShotRequest:
    request = NativeShotRequest(
        shot_id="shot-action",
        scene_id="scene-action",
        camera={
            "resolution": (1280, 704),
            "fps": 24,
            "duration": 5.0,
            "shot_size": "full-body two-shot",
            "movement": "fast orbit tracking",
            "lens": "35mm anamorphic",
        },
        characters=[],
        environment={
            "name": "rainy loading dock",
            "lighting": "hard sodium practicals changing to emergency red",
        },
        wardrobe=[{"character_uuid": "hero", "lock": "black field jacket"}],
        props=[{"prop_id": "case", "continuity": "held in hero right hand"}],
        continuity={"previous_shot_id": "shot-before", "screen_direction": "left-to-right"},
        performance={
            "blocking": "hero runs beside partner, catches the case, then turns",
            "object_interaction": "case transfers from partner left hand to hero right hand",
            "dialogue_timing": [
                {"speaker_id": "hero", "start_frame": 36, "end_frame": 66, "text": "Move!"}
            ],
        },
        approved_reference_ids=[],
        deterministic_seed=17,
        renderer_requirements={},
        metadata={"prompt": "Two characters escape the loading dock."},
    )
    request.refresh_hash()
    return request


def test_director_prompt_preserves_structured_performance_and_object_interaction():
    prompt = ProductionDiffusersVideoRenderer._compile_prompt(_request())

    assert prompt.startswith(
        "Two characters escape the loading dock.\nCINEOS production constraints"
    )
    assert '"blocking":"hero runs beside partner, catches the case, then turns"' in prompt
    assert '"object_interaction":"case transfers from partner left hand to hero right hand"' in prompt
    assert '"dialogue_timing"' in prompt
    assert '"prop_id":"case"' in prompt
    assert '"held in hero right hand"' in prompt


def test_director_prompt_preserves_environment_wardrobe_and_camera_choreography():
    prompt = ProductionDiffusersVideoRenderer._compile_prompt(_request())

    assert '"environment"' in prompt
    assert '"hard sodium practicals changing to emergency red"' in prompt
    assert '"black field jacket"' in prompt
    assert '"movement":"fast orbit tracking"' in prompt
    assert '"shot_size":"full-body two-shot"' in prompt
    assert '"lens":"35mm anamorphic"' in prompt
    assert '"screen_direction":"left-to-right"' in prompt
